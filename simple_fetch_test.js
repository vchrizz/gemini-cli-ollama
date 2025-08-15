#!/usr/bin/env node

// Simple fetch test to debug the hanging issue
console.log('🧪 Testing direct fetch to Ollama...');

const request = {
  model: "gpt-oss:20b",
  stream: false,
  messages: [
    { role: "system", content: "You are a helpful assistant with access to tools." },
    { role: "user", content: "Use the shell tool to run the command 'uptime' to check system uptime." }
  ],
  tools: [{
    type: "function",
    function: {
      name: "run_shell_command",
      description: "Execute a shell command",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "The shell command to execute" }
        },
        required: ["command"]
      }
    }
  }]
};

console.log('📤 Request:', JSON.stringify(request, null, 2));

try {
  console.log('🌐 Making fetch request...');
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    console.log('⏰ Timeout reached, aborting...');
    controller.abort();
  }, 10000); // 10 second timeout
  
  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: controller.signal
  });
  
  clearTimeout(timeoutId);
  console.log('📡 Response status:', response.status, response.statusText);
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error('❌ API error:', errorText);
    process.exit(1);
  }
  
  console.log('🔄 Reading response text...');
  const responseText = await response.text();
  console.log('📄 Response length:', responseText.length);
  console.log('📄 Response:', responseText);
  
  console.log('✅ Test completed successfully');
  
} catch (error) {
  console.error('❌ Error:', error.message);
  if (error.name === 'AbortError') {
    console.error('❌ Request was aborted due to timeout');
  }
  process.exit(1);
}