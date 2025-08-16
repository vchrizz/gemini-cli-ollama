#!/usr/bin/env node

// Test if large requests hang Ollama

async function testLargeRequest() {
  // Create a progressively larger request to find the breaking point
  const sizes = [1000, 5000, 10000, 20000, 50000, 100000];
  
  for (const size of sizes) {
    console.log(`\n🧪 Testing request size: ${size} characters`);
    
    const largeContent = 'This is test content. '.repeat(Math.floor(size / 22));
    const actualSize = largeContent.length;
    
    const request = {
      model: "gpt-oss:20b",
      stream: false,
      messages: [
        {role: "user", content: largeContent}
      ]
    };
    
    const requestJson = JSON.stringify(request);
    console.log(`📏 Actual request size: ${requestJson.length} bytes`);
    
    try {
      const startTime = Date.now();
      const response = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: requestJson,
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });
      
      const elapsed = Date.now() - startTime;
      
      if (!response.ok) {
        console.log(`❌ HTTP ${response.status}: ${response.statusText} (${elapsed}ms)`);
        const errorText = await response.text();
        console.log(`Error: ${errorText.substring(0, 200)}...`);
        break;
      }
      
      const result = await response.json();
      console.log(`✅ Success in ${elapsed}ms - Response: "${result.message?.content?.substring(0, 50)}..."`);
      
    } catch (error) {
      const elapsed = Date.now() - startTime;
      if (error.name === 'TimeoutError') {
        console.log(`⏰ TIMEOUT after ${elapsed}ms - Request likely hanging!`);
        break;
      } else {
        console.log(`❌ Error after ${elapsed}ms: ${error.message}`);
        break;
      }
    }
  }
}

testLargeRequest().catch(console.error);