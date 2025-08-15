#!/usr/bin/env node

// Direct test of our OllamaContentGenerator with the exact same request as the working cURL

import { OllamaContentGenerator } from './packages/core/dist/src/core/ollamaContentGenerator.js';

const config = {
  baseUrl: 'http://localhost:11434',
  model: 'gpt-oss:20b',
  timeout: 120,
  contextLimit: 2048,
  enableChatApi: true
};

console.log('Testing OllamaContentGenerator with tool calling...');
console.log('Config:', config);

const generator = new OllamaContentGenerator(config);

// Exact same request as the working cURL
const request = {
  contents: [{
    role: 'user',
    parts: [{
      text: "Use the shell tool to run the command 'uptime' to check system uptime."
    }]
  }],
  systemInstruction: {
    role: 'system',
    parts: [{
      text: "You are a helpful assistant with access to tools."
    }]
  },
  tools: [{
    functionDeclarations: [{
      name: "run_shell_command",
      description: "Execute a shell command",
      parameters: {
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
  }],
  generationConfig: {
    temperature: 0.7
  }
};

console.log('Request:', JSON.stringify(request, null, 2));

try {
  console.log('\nStarting request...');
  const startTime = Date.now();
  
  const result = await generator.generateContent(request);
  
  const duration = Date.now() - startTime;
  console.log(`✅ Request completed in ${duration}ms`);
  console.log('Result:', JSON.stringify(result, null, 2));
  
} catch (error) {
  console.error('❌ Request failed:', error.message);
  console.error('Stack:', error.stack);
}