"""
Demo Advanced Virtual Try-On System

This script demonstrates the advanced virtual try-on system with:
- Body shape analysis
- Realistic clothing fitting
- Quality assessment
- Multiple clothing types

Author: Personal AI Stylist System
Date: 2025-01-27
"""

import os
import sys
from pathlib import Path
import logging
import time

# Import our advanced try-on system
from advanced_tryon import AdvancedTryOnEngine, ClothingType, BodyShape

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def demo_advanced_tryon():
    """Demo the advanced virtual try-on system"""
    print("🎭 ADVANCED VIRTUAL TRY-ON DEMO")
    print("=" * 50)
    
    # Initialize the advanced try-on engine
    print("🚀 Initializing Advanced Try-On Engine...")
    engine = AdvancedTryOnEngine()
    print("✅ Engine initialized successfully!")
    
    # Check for available images
    person_dir = Path("data/user_photos")
    clothing_dir = Path("data/wardrobe_items")
    
    if not person_dir.exists():
        print("❌ Person images directory not found: data/user_photos/")
        print("📁 Please add your photos to this directory")
        return
    
    if not clothing_dir.exists():
        print("❌ Clothing images directory not found: data/wardrobe_items/")
        print("📁 Please add clothing images to this directory")
        return
    
    # Find available images
    person_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))
    clothing_images = list(clothing_dir.glob("*.jpg")) + list(clothing_dir.glob("*.jpeg")) + list(clothing_dir.glob("*.png"))
    
    if not person_images:
        print("❌ No person images found in data/user_photos/")
        print("📁 Please add some photos to test the system")
        return
    
    if not clothing_images:
        print("❌ No clothing images found in data/wardrobe_items/")
        print("📁 Please add some clothing images to test the system")
        return
    
    print(f"\n📸 Found {len(person_images)} person image(s):")
    for img in person_images:
        print(f"   • {img.name}")
    
    print(f"\n👕 Found {len(clothing_images)} clothing item(s):")
    for img in clothing_images:
        print(f"   • {img.name}")
    
    # Test with first available images
    person_image = str(person_images[0])
    clothing_image = str(clothing_images[0])
    
    print(f"\n🎯 Testing with:")
    print(f"   Person: {Path(person_image).name}")
    print(f"   Clothing: {Path(clothing_image).name}")
    
    # Test different clothing types
    clothing_types = [
        ClothingType.SHIRT,
        ClothingType.T_SHIRT,
        ClothingType.SWEATER
    ]
    
    results = []
    
    for clothing_type in clothing_types:
        print(f"\n{'='*60}")
        print(f"🔄 Testing {clothing_type.value.upper()} fitting...")
        
        try:
            start_time = time.time()
            
            # Perform advanced try-on
            result = engine.try_on(person_image, clothing_image, clothing_type)
            
            processing_time = time.time() - start_time
            
            if result.success:
                print(f"✅ {clothing_type.value.title()} try-on successful!")
                print(f"   📊 Body Shape: {result.body_measurements.body_shape.value.title()}")
                print(f"   🎯 Fitting Quality: {result.fitting_quality:.2f}")
                print(f"   ⏱️  Processing Time: {result.processing_time:.2f}s")
                print(f"   📁 Result saved to: {result.result_image_path}")
                
                results.append({
                    'type': clothing_type.value,
                    'success': True,
                    'quality': result.fitting_quality,
                    'time': result.processing_time,
                    'body_shape': result.body_measurements.body_shape.value,
                    'confidence': result.body_measurements.confidence
                })
            else:
                print(f"❌ {clothing_type.value.title()} try-on failed!")
                print(f"   Error: {result.error_message}")
                
                results.append({
                    'type': clothing_type.value,
                    'success': False,
                    'error': result.error_message
                })
                
        except Exception as e:
            print(f"❌ Error during {clothing_type.value} try-on: {e}")
            results.append({
                'type': clothing_type.value,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n🎊 DEMO COMPLETE!")
    print("=" * 50)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n🏆 Best Results:")
        # Sort by quality
        successful.sort(key=lambda x: x['quality'], reverse=True)
        
        for i, result in enumerate(successful[:3], 1):
            print(f"   {i}. {result['type'].title()}: Quality {result['quality']:.2f}, "
                  f"Body Shape: {result['body_shape'].title()}")
    
    if failed:
        print(f"\n⚠️ Failed Attempts:")
        for result in failed:
            print(f"   • {result['type'].title()}: {result['error']}")
    
    print(f"\n📁 All results saved in: output/advanced_tryon/")
    print(f"🎭 Try the Virtual Try-On Room: python virtual_tryon_room.py")


def demo_body_analysis():
    """Demo body analysis capabilities"""
    print("\n🔍 BODY ANALYSIS DEMO")
    print("=" * 30)
    
    try:
        from advanced_tryon import AdvancedBodyAnalyzer
        
        # Initialize body analyzer
        analyzer = AdvancedBodyAnalyzer()
        
        # Find a person image
        person_dir = Path("data/user_photos")
        person_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))
        
        if person_images:
            person_image = str(person_images[0])
            print(f"📸 Analyzing: {Path(person_image).name}")
            
            # Analyze body
            measurements, segmentation_mask, metadata = analyzer.analyze_body(person_image)
            
            print(f"✅ Body analysis complete!")
            print(f"   📊 Body Shape: {measurements.body_shape.value.title()}")
            print(f"   🎯 Confidence: {measurements.confidence:.2f}")
            print(f"   📏 Shoulder Width: {measurements.shoulder_width:.1f}px")
            print(f"   📏 Chest Width: {measurements.chest_width:.1f}px")
            print(f"   📏 Waist Width: {measurements.waist_width:.1f}px")
            print(f"   📏 Torso Height: {measurements.torso_height:.1f}px")
            print(f"   🖼️ Image Size: {metadata['image_size']}")
            print(f"   🎭 Segmentation Quality: {metadata['segmentation_quality']:.2f}")
            
        else:
            print("❌ No person images found for body analysis")
            
    except Exception as e:
        print(f"❌ Body analysis demo failed: {e}")


def main():
    """Main demo function"""
    print("🎭 PERSONAL AI STYLIST - ADVANCED VIRTUAL TRY-ON DEMO")
    print("=" * 60)
    
    # Check if required directories exist
    required_dirs = ["data/user_photos", "data/wardrobe_items"]
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("⚠️ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   • {dir_path}")
        print("\n📁 Please create these directories and add some images:")
        print("   • data/user_photos/ - Add your photos here")
        print("   • data/wardrobe_items/ - Add clothing images here")
        return
    
    # Run demos
    demo_body_analysis()
    demo_advanced_tryon()
    
    print(f"\n✨ Demo complete! Check the results in output/advanced_tryon/")
    print(f"🎭 For interactive experience, run: python virtual_tryon_room.py")


if __name__ == "__main__":
    main()
