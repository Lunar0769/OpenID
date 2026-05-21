"""
Complete Camera Capture Test Suite
Test both passport and ID card capture flows
"""
from openid import OpenIDClient
from openid.flows.passport import capture_passport
from openid.flows.id_card import capture_id_card

def main():
    # Initialize client with production API
    client = OpenIDClient(api_key="your_key", timeout=120)
    
    print("\n" + "="*60)
    print("OpenID Camera Capture Test Suite")
    print("="*60)
    print("\nSelect document type to capture:")
    print("  1. Passport")
    print("  2. Emirates ID")
    print("  3. Driving License")
    print("  4. ID Card (Auto-detect)")
    print("  0. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (0-4): ").strip()
    
    if choice == "1":
        print("\n🛂 Starting Passport Capture...")
        result = capture_passport(client)
        
    elif choice == "2":
        print("\n🪪 Starting Emirates ID Capture...")
        result = capture_id_card(client, doc_type="emirates_id")
        
    elif choice == "3":
        print("\n🚗 Starting Driving License Capture...")
        result = capture_id_card(client, doc_type="driving_license")
        
    elif choice == "4":
        print("\n🪪 Starting ID Card Capture (Auto-detect)...")
        result = capture_id_card(client, doc_type="auto")
        
    elif choice == "0":
        print("\n👋 Exiting...")
        return
        
    else:
        print("\n❌ Invalid choice. Please run again and select 0-4.")
        return
    
    # Display result summary
    if result:
        print("\n" + "="*60)
        print("✅ Extraction completed successfully!")
        print("="*60)
        if "data" in result:
            print("\n📋 Extracted Fields:")
            data = result.get("data", {})
            ocr_data = data.get("ocrData", {})
            for key, value in ocr_data.items():
                if value and key not in ["notExtracted", "ocrDataConfidence"]:
                    print(f"  • {key}: {value}")
    else:
        print("\n" + "="*60)
        print("❌ Capture was cancelled or failed.")
        print("="*60)

if __name__ == "__main__":
    main()
