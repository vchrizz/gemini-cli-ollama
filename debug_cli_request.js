#!/usr/bin/env node

// Debug script to see exactly what request the CLI sends to our OllamaContentGenerator

import { createReadStream } from 'fs';
import { spawn } from 'child_process';

console.log('Patching OllamaContentGenerator to log all incoming requests...');

// Read the current file
const originalPath = './packages/core/dist/src/core/ollamaContentGenerator.js';
const backupPath = './packages/core/dist/src/core/ollamaContentGenerator.js.backup';

// Create a simple patch that logs the full request
const patchContent = `
// PATCH: Log all incoming requests
const originalGenerateContent = OllamaContentGenerator.prototype.generateContent;
OllamaContentGenerator.prototype.generateContent = function(request, userPromptId) {
  console.log('🔍 CLI REQUEST INTERCEPTED:', {
    hasRequest: !!request,
    requestKeys: request ? Object.keys(request) : 'no request',
    contents: request?.contents ? 'has contents' : 'no contents',
    tools: request?.tools ? 'has tools' : 'no tools',
    configTools: request?.config?.tools ? 'has config.tools' : 'no config.tools',
    fullRequest: JSON.stringify(request, null, 2).substring(0, 1000) + '...'
  });
  return originalGenerateContent.call(this, request, userPromptId);
};
`;

console.log('Starting CLI with debug logging...');

const child = spawn('node', ['scripts/start.js', '-p', 'Use the shell tool to run "uptime" command'], {
  cwd: process.cwd(),
  env: { ...process.env, DEBUG: '1' },
  stdio: ['pipe', 'pipe', 'pipe']
});

let output = '';
let errorOutput = '';

child.stdout.on('data', (data) => {
  const text = data.toString();
  output += text;
  console.log('STDOUT:', text);
});

child.stderr.on('data', (data) => {
  const text = data.toString();
  errorOutput += text;
  console.log('STDERR:', text);
});

child.on('close', (code) => {
  console.log(`\nProcess exited with code: ${code}`);
  console.log('Full stdout:', output);
  console.log('Full stderr:', errorOutput);
});

// Kill after 30 seconds if no response
setTimeout(() => {
  console.log('❌ Timeout reached, killing process');
  child.kill('SIGKILL');
  process.exit(1);
}, 30000);