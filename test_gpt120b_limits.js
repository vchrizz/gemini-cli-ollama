#!/usr/bin/env node

// Test exact limits for gpt-oss:120b (large model)
async function testGpt120bLimits() {
  console.log('🧪 Testing gpt-oss:120b exact limits...');
  console.log('📝 This is the largest model - expecting potentially higher limits');
  
  // Test various sizes for the 120B model - might handle larger requests
  const testSizes = [20, 30, 40, 50, 75, 100, 150, 200, 300]; // KB
  
  for (const sizeKB of testSizes) {
    const targetSize = sizeKB * 1000;
    const largeContent = 'x'.repeat(targetSize);
    
    const request = {
      model: "gpt-oss:120b",
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
        signal: AbortSignal.timeout(120000) // 2 minute timeout for large model
      });
      
      const elapsed = Date.now() - startTime;
      
      if (!response.ok) {
        console.log(`❌ FAILED: HTTP ${response.status} (${elapsed}ms)`);
        const errorText = await response.text();
        console.log(`   Error: ${errorText.substring(0, 100)}...`);
        
        if (response.status === 500) {
          console.log(`💡 120B model limit found at ${sizeKB}KB`);
        }
        break; // Stop at first failure
      } else {
        const result = await response.json();
        console.log(`✅ SUCCESS: ${elapsed}ms - Response length: ${result.message?.content?.length || 0} chars`);
        
        // Show performance metrics for large model
        if (elapsed > 30000) {
          console.log(`⚠️  Very slow response (${(elapsed/1000).toFixed(1)}s) - likely due to model size`);
        }
      }
      
    } catch (error) {
      const elapsed = Date.now() - startTime;
      if (error.name === 'TimeoutError') {
        console.log(`⏰ TIMEOUT: ${elapsed}ms - Possible hanging or very slow processing`);
        console.log(`💡 120B model may need longer timeout or smaller requests`);
        break;
      } else {
        console.log(`❌ ERROR: ${error.message}`);
        break;
      }
    }
    
    // Longer delay between requests for large model
    console.log('⏳ Waiting 5 seconds before next test...');
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  
  console.log('\n📊 Test Summary for gpt-oss:120b:');
  console.log('- This tests the largest available model');
  console.log('- May have different limits than smaller models');
  console.log('- Useful for understanding scaling behavior');
}

testGpt120bLimits().catch(console.error);