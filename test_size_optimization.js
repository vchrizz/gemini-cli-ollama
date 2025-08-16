#!/usr/bin/env node

// Test the new request size optimization 

async function testSizeOptimization() {
  console.log('🧪 Testing request size optimization...');
  
  // Create a 25KB request (larger than our 18KB limit) to test truncation
  const largeContent = 'This is a very long test message. '.repeat(750); // ~25KB
  const actualSize = largeContent.length;
  
  console.log(`📏 Original content size: ${actualSize} chars (~${(actualSize/1000).toFixed(1)}KB)`);
  console.log(`📏 Expected: Content should be truncated to ~18KB limit`);
  
  const request = {
    model: "qwen3:30b",
    stream: false,
    messages: [{role: "user", content: largeContent}]
  };
  
  const requestJson = JSON.stringify(request);
  console.log(`📏 Full request size: ${requestJson.length} chars (~${(requestJson.length/1000).toFixed(1)}KB)`);
  
  const startTime = Date.now();
  try {
    const response = await fetch('http://localhost:11434/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: requestJson,
      signal: AbortSignal.timeout(60000) // 1 minute timeout
    });
    
    const elapsed = Date.now() - startTime;
    
    if (!response.ok) {
      console.log(`❌ FAILED: HTTP ${response.status} (${elapsed}ms)`);
      const errorText = await response.text();
      console.log(`   Error: ${errorText.substring(0, 200)}...`);
      
      if (response.status === 500) {
        console.log(`\n💡 This confirms the issue: large requests still cause model crashes`);
        console.log(`💡 The optimization should prevent this by truncating content in the Gemini CLI`);
      }
    } else {
      const result = await response.json();
      console.log(`✅ SUCCESS: ${elapsed}ms - Response length: ${result.message?.content?.length || 0} chars`);
      console.log(`📋 Response preview: "${result.message?.content?.substring(0, 100)}..."`);
    }
    
  } catch (error) {
    const elapsed = Date.now() - startTime;
    if (error.name === 'TimeoutError') {
      console.log(`⏰ TIMEOUT: ${elapsed}ms - Possible hanging`);
    } else {
      console.log(`❌ ERROR: ${error.message}`);
    }
  }
  
  console.log('\n📝 Note: This test shows the raw Ollama API behavior.');
  console.log('📝 The Gemini CLI now truncates content before sending to Ollama.');
  console.log('📝 To test the CLI optimization, use the actual gemini CLI with large context.');
}

testSizeOptimization().catch(console.error);