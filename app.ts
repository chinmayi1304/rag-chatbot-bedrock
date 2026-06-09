#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { RagStack } from "../lib/rag-stack";

const app = new cdk.App();

new RagStack(app, "RagChatbotStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "us-east-1",
  },
  tags: {
    Project: "rag-chatbot-bedrock",
    Owner: "Chinmayi",
    Environment: "dev",
  },
});
