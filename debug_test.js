#!/usr/bin/env node

// Simple direct test of OllamaContentGenerator
import { OllamaContentGenerator } from './dist/packages/core/src/core/ollamaContentGenerator.js';

const config = {
  baseUrl: 'http://localhost:11434',
  model: 'gpt-oss:20b',
  enableChatApi: true,
  timeout: 10000, // 10 seconds
  contextLimit: 2048,
  debugMode: true
};

const generator = new OllamaContentGenerator(config);

const request = {
  contents: "Use the shell tool to run uptime command",
  config: {
    tools: [{
      functionDeclarations: [{
        name: "run_shell_command",
        description: "Execute a shell command",
        parametersJsonSchema: {
          type: "object",
          properties: {
            command: {
              type: "string",
              description: "The shell command to execute"
            }
          },
          required: ["command"]
        }
      }]
    }]
  }
};

console.log('🚀 Starting direct OllamaContentGenerator test...');

try {
  const result = await generator.generateContent(request, 'test-prompt-id');
  console.log('✅ Success:', result);
} catch (error) {
  console.error('❌ Error:', error.message);
  console.error('Stack:', error.stack);
}