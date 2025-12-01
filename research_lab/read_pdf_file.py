"""Read a PDF file using the PDF reader tool."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.pdf_reader import PDFReaderTool

def read_pdf_file(pdf_path: str):
    """Read and display PDF content."""
    print("=" * 60)
    print(f"Reading PDF: {pdf_path}")
    print("=" * 60)
    
    try:
        # Initialize PDF reader
        print("\n1️⃣  Initializing PDF Reader...")
        tool = PDFReaderTool()
        print("   ✅ PDF Reader initialized!")
        
        # Read PDF
        print(f"\n2️⃣  Reading PDF file...")
        print(f"   Path: {pdf_path}")
        result = tool.read_pdf(pdf_path, max_pages=100, start_page=1)
        
        if result["success"]:
            print(f"   ✅ Success!")
            print(f"   📊 Total pages: {result['total_pages']}")
            print(f"   📊 Extracted pages: {result['extracted_pages']}")
            print(f"   📊 Pages range: {result['start_page']}-{result['end_page']}")
            
            text = result["text"]
            print(f"\n3️⃣  Content Preview (first 3000 characters):")
            print("   " + "-" * 56)
            print("   " + text[:3000].replace("\n", "\n   "))
            if len(text) > 3000:
                print("   ...")
            print("   " + "-" * 56)
            
            print(f"\n4️⃣  Full Content Statistics:")
            print(f"   Total characters: {len(text):,}")
            print(f"   Total words: ~{len(text.split()):,}")
            
            # Save to file for easier viewing
            output_file = "pdf_content_output.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"PDF: {pdf_path}\n")
                f.write(f"Pages: {result['total_pages']} (extracted {result['extracted_pages']})\n")
                f.write("=" * 80 + "\n\n")
                f.write(text)
            
            print(f"\n   💾 Full content saved to: {output_file}")
            
            return True
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # PDF path relative to research_lab directory
    pdf_path = "../2409.05556v1.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    success = read_pdf_file(pdf_path)
    sys.exit(0 if success else 1)

