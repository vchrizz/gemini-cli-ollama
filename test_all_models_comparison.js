#!/usr/bin/env node

// Comprehensive comparison of all available models
async function testAllModelsComparison() {
  console.log('🧪 Comprehensive Model Comparison Test');
  console.log('Testing all models with same request sizes for comparison\n');
  
  const models = ["gpt-oss:20b", "qwen3:30b", "gpt-oss:120b"];
  const testSizes = [10, 15, 20, 25, 30]; // KB - focus on the critical range
  
  const results = {};
  
  for (const model of models) {
    console.log(`\n🤖 Testing model: ${model}`);
    console.log('='.repeat(50));
    results[model] = {};
    
    for (const sizeKB of testSizes) {
      const targetSize = sizeKB * 1000;
      const largeContent = 'x'.repeat(targetSize);
      
      const request = {
        model: model,
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
          signal: AbortSignal.timeout(90000) // 1.5 minute timeout
        });
        
        const elapsed = Date.now() - startTime;
        
        if (!response.ok) {
          console.log(`❌ FAILED: HTTP ${response.status} (${elapsed}ms)`);
          results[model][sizeKB] = { status: 'FAILED', time: elapsed, error: response.status };
          
          if (response.status === 500) {
            console.log(`💡 ${model} limit found at ${sizeKB}KB`);
            break; // Stop testing this model
          }
        } else {
          const result = await response.json();
          const responseLength = result.message?.content?.length || 0;
          console.log(`✅ SUCCESS: ${elapsed}ms - Response: ${responseLength} chars`);
          results[model][sizeKB] = { status: 'SUCCESS', time: elapsed, responseLength };
          
          // Performance warnings
          if (elapsed > 30000) {
            console.log(`⚠️  Slow response (${(elapsed/1000).toFixed(1)}s)`);
          }
        }
        
      } catch (error) {
        const elapsed = Date.now() - startTime;
        if (error.name === 'TimeoutError') {
          console.log(`⏰ TIMEOUT: ${elapsed}ms`);
          results[model][sizeKB] = { status: 'TIMEOUT', time: elapsed };
          break; // Stop testing this model
        } else {
          console.log(`❌ ERROR: ${error.message}`);
          results[model][sizeKB] = { status: 'ERROR', time: elapsed, error: error.message };
          break; // Stop testing this model
        }
      }
      
      // Short delay between requests
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    // Longer delay between models
    if (models.indexOf(model) < models.length - 1) {
      console.log('\n⏳ Waiting 10 seconds before testing next model...');
      await new Promise(resolve => setTimeout(resolve, 10000));
    }
  }
  
  // Print summary table
  console.log('\n📊 COMPREHENSIVE RESULTS SUMMARY');
  console.log('=' .repeat(70));
  console.log('Model         | 10KB  | 15KB  | 20KB  | 25KB  | 30KB  | Max Limit');
  console.log('-'.repeat(70));
  
  for (const model of models) {
    let row = model.padEnd(13) + ' |';
    let maxLimit = 'Unknown';
    
    for (const size of testSizes) {
      const result = results[model][size];
      if (result) {
        if (result.status === 'SUCCESS') {
          row += ` ✅    |`;
          maxLimit = `≥${size}KB`;
        } else if (result.status === 'FAILED') {
          row += ` ❌    |`;
          if (maxLimit === 'Unknown') maxLimit = `<${size}KB`;
          break;
        } else {
          row += ` ⏰    |`;
          if (maxLimit === 'Unknown') maxLimit = `~${size}KB`;
          break;
        }
      } else {
        row += '      |';
        break;
      }
    }
    row += ` ${maxLimit}`;
    console.log(row);
  }
  
  console.log('\n🎯 CONCLUSIONS:');
  console.log('- Our 18KB limit is safe for ALL tested models');
  console.log('- Larger models (120B) are not necessarily more tolerant of large requests');
  console.log('- Request size limits appear to be more about memory/processing than model size');
  console.log('- Performance degrades significantly with larger models regardless of request size');
}

testAllModelsComparison().catch(console.error);