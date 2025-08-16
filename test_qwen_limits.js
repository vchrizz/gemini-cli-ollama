#!/usr/bin/env node

// Test exact limits for qwen3:30b
async function testQwenLimits() {
  console.log('🧪 Testing qwen3:30b exact limits...');
  
  // Test larger sizes for qwen3:30b
  const testSizes = [30, 40, 50, 75, 100, 150, 200]; // KB
  
  for (const sizeKB of testSizes) {
    const targetSize = sizeKB * 1000;
    const largeContent = 'x'.repeat(targetSize);
    
    const request = {
      model: "qwen3:30b",
      stream: false,
      messages: [{role: "user", content: largeContent}]
    };
    
    const requestJson = JSON.stringify(request);
    const actualSize = requestJson.length;
    
    console.log(`\n📏 Testing ${sizeKB}KB (actual: ${(actualSize/1000).toFixed(1)}KB)`);
    
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
        console.log(`   Error: ${errorText.substring(0, 100)}...`);
        break; // Stop at first failure
      } else {
        const result = await response.json();
        console.log(`✅ SUCCESS: ${elapsed}ms - Response length: ${result.message?.content?.length || 0} chars`);
      }
      
    } catch (error) {
      const elapsed = Date.now() - startTime;
      if (error.name === 'TimeoutError') {
        console.log(`⏰ TIMEOUT: ${elapsed}ms - Possible hanging`);
        break;
      } else {
        console.log(`❌ ERROR: ${error.message}`);
        break;
      }
    }
    
    // Delay between requests
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

testQwenLimits().catch(console.error);