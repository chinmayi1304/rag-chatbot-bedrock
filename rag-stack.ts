import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as apigateway from "aws-cdk-lib/aws-apigateway";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3n from "aws-cdk-lib/aws-s3-notifications";
import { Construct } from "constructs";
import * as path from "path";

export class RagStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ─── 1. S3 bucket — stores uploaded PDFs ────────────────────────────
    // Free tier: 5 GB storage, 20k GET, 2k PUT requests per month (12 months)
    const docsBucket = new s3.Bucket(this, "DocsBucket", {
      bucketName: `rag-docs-${this.account}-${this.region}`,
      cors: [
        {
          allowedMethods: [s3.HttpMethods.PUT, s3.HttpMethods.POST],
          allowedOrigins: ["*"],
          allowedHeaders: ["*"],
        },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    });

    // ─── 2. DynamoDB — stores document metadata ──────────────────────────
    // Tracks: docId, filename, chunk count, upload timestamp, status
    // Free tier: 25 GB storage, 25 RCU, 25 WCU — more than enough
    const docsTable = new dynamodb.Table(this, "DocsTable", {
      tableName: "rag-documents",
      partitionKey: { name: "docId", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ─── 3. VPC for EC2 (ChromaDB) ───────────────────────────────────────
    // Using default VPC to avoid NAT gateway costs
    const vpc = ec2.Vpc.fromLookup(this, "DefaultVpc", { isDefault: true });

    // Security group — only allow Lambda to reach ChromaDB on port 8000
    const chromaSecurityGroup = new ec2.SecurityGroup(this, "ChromaSG", {
      vpc,
      description: "Allow Lambda to reach ChromaDB",
      allowAllOutbound: true,
    });
    chromaSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8000),
      "ChromaDB API port"
    );
    chromaSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      "SSH for debugging"
    );

    // ─── 4. EC2 t2.micro — runs ChromaDB ─────────────────────────────────
    // Free tier: 750 hours/month for 12 months — runs 24/7 for free
    // UserData installs Docker + ChromaDB automatically on first boot
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      "#!/bin/bash",
      "yum update -y",
      "yum install -y docker",
      "systemctl start docker",
      "systemctl enable docker",
      "usermod -a -G docker ec2-user",
      // Run ChromaDB in Docker, persist data to /chroma-data
      "mkdir -p /chroma-data",
      "docker run -d \\",
      "  --name chromadb \\",
      "  --restart always \\",
      "  -p 8000:8000 \\",
      "  -v /chroma-data:/chroma/chroma \\",
      "  -e IS_PERSISTENT=TRUE \\",
      "  chromadb/chroma:latest",
      // Confirm it started
      'echo "ChromaDB started" >> /var/log/user-data.log'
    );

    const chromaInstance = new ec2.Instance(this, "ChromaInstance", {
      vpc,
      instanceType: ec2.InstanceType.of(
        ec2.InstanceClass.T2,
        ec2.InstanceSize.MICRO
      ),
      machineImage: ec2.MachineImage.latestAmazonLinux2(),
      securityGroup: chromaSecurityGroup,
      userData,
    });

    // ─── 5. IAM role for Lambdas — scoped Bedrock access ─────────────────
    const lambdaRole = new iam.Role(this, "LambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AWSLambdaBasicExecutionRole"
        ),
      ],
    });

    // Bedrock: allow both Titan Embeddings and Claude Haiku
    lambdaRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:InvokeModel"],
        resources: [
          "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-20240307-v1:0",
        ],
      })
    );

    docsBucket.grantReadWrite(lambdaRole);
    docsTable.grantWriteData(lambdaRole);
    docsTable.grantReadData(lambdaRole);

    // ─── 6. Lambda: Ingestor ─────────────────────────────────────────────
    // Triggered by S3 event when a PDF is uploaded
    // Extracts text, chunks it, gets Titan embeddings, stores in ChromaDB
    const ingestorFn = new lambda.Function(this, "IngestorFunction", {
      functionName: "rag-ingestor",
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "handler.lambda_handler",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "../../lambdas/ingestor"),
        {
          bundling: {
            image: lambda.Runtime.PYTHON_3_12.bundlingImage,
            command: [
              "bash", "-c",
              "pip install -r requirements.txt -t /asset-output && cp -r . /asset-output",
            ],
          },
        }
      ),
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      role: lambdaRole,
      environment: {
        DOCS_TABLE: docsTable.tableName,
        CHROMA_HOST: chromaInstance.instancePublicIp,
        CHROMA_PORT: "8000",
        CHROMA_COLLECTION: "rag-docs",
        EMBED_MODEL_ID: "amazon.titan-embed-text-v1",
        REGION: this.region,
      },
    });

    // Trigger ingestor when a PDF is uploaded to S3
    docsBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.LambdaDestination(ingestorFn),
      { suffix: ".pdf" }
    );

    // ─── 7. Lambda: Query ────────────────────────────────────────────────
    // Receives user question from API Gateway
    // Embeds query → searches ChromaDB → builds prompt → calls Claude Haiku
    const queryFn = new lambda.Function(this, "QueryFunction", {
      functionName: "rag-query",
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "handler.lambda_handler",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "../../lambdas/query"),
        {
          bundling: {
            image: lambda.Runtime.PYTHON_3_12.bundlingImage,
            command: [
              "bash", "-c",
              "pip install -r requirements.txt -t /asset-output && cp -r . /asset-output",
            ],
          },
        }
      ),
      timeout: cdk.Duration.seconds(60),
      memorySize: 256,
      role: lambdaRole,
      environment: {
        DOCS_TABLE: docsTable.tableName,
        CHROMA_HOST: chromaInstance.instancePublicIp,
        CHROMA_PORT: "8000",
        CHROMA_COLLECTION: "rag-docs",
        EMBED_MODEL_ID: "amazon.titan-embed-text-v1",
        // Haiku is the cheapest Claude model — ~$0.25/1M input tokens
        // On Bedrock free trial you get 1M tokens/month free
        LLM_MODEL_ID: "anthropic.claude-haiku-20240307-v1:0",
        REGION: this.region,
        TOP_K_CHUNKS: "4",
      },
    });

    // ─── 8. API Gateway ───────────────────────────────────────────────────
    const api = new apigateway.RestApi(this, "RagApi", {
      restApiName: "RAG Chatbot API",
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: ["POST", "GET", "OPTIONS"],
        allowHeaders: ["Content-Type"],
      },
    });

    // POST /query  — ask a question
    const queryResource = api.root.addResource("query");
    queryResource.addMethod(
      "POST",
      new apigateway.LambdaIntegration(queryFn)
    );

    // POST /ingest — trigger manual ingest (optional, S3 trigger is automatic)
    const ingestResource = api.root.addResource("ingest");
    ingestResource.addMethod(
      "POST",
      new apigateway.LambdaIntegration(ingestorFn)
    );

    // ─── 9. Outputs ───────────────────────────────────────────────────────
    new cdk.CfnOutput(this, "ApiUrl", {
      value: api.url,
      description: "Paste into frontend/app.js as API_URL",
    });
    new cdk.CfnOutput(this, "BucketName", {
      value: docsBucket.bucketName,
      description: "Upload PDFs to this S3 bucket",
    });
    new cdk.CfnOutput(this, "ChromaInstanceIp", {
      value: chromaInstance.instancePublicIp,
      description: "ChromaDB EC2 public IP — wait 2 min after deploy for Docker to start",
    });
  }
}
