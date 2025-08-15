#!/usr/bin/env node

// Test script to check if the tool calling implementation works properly
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

// Test by running the CLI with a simple tool calling prompt
const testPrompt = 'Use the shell tool to run "uptime" command to check system uptime.';

console.log('Testing Ollama tool calling implementation...');
console.log('Prompt:', testPrompt);

// Set up environment
const env = {
  ...process.env,
  DEBUG: '1', // Enable debug mode to see what's happening
};

const child = spawn('node', ['scripts/start.js', '-p', testPrompt], {
  cwd: process.cwd(),
  env: env,
  stdio: ['pipe', 'pipe', 'pipe']
});

let stdout = '';
let stderr = '';

child.stdout.on('data', (data) => {
  const text = data.toString();
  stdout += text;
  process.stdout.write(text);
});

child.stderr.on('data', (data) => {
  const text = data.toString();
  stderr += text;
  process.stderr.write(text);
});

child.on('close', (code) => {
  console.log(`\n\nProcess exited with code: ${code}`);
  
  if (code === 0) {
    console.log('✅ Test passed - CLI executed successfully');
  } else {
    console.log('❌ Test failed - CLI execution failed');
    console.log('STDOUT:', stdout);
    console.log('STDERR:', stderr);
  }
});

// Timeout after 2 minutes
setTimeout(() => {
  console.log('❌ Test timed out - killing process');
  child.kill('SIGKILL');
  process.exit(1);
}, 120000);