# 🌍 Climate Heroes KPI Extractor

An AI-powered tool to automatically extract sustainability metrics from PDF reports using Google's Gemini AI. Perfect for ESG analysts, sustainability professionals, and researchers who need to process multiple sustainability reports efficiently.

## ✨ Features

- **Batch Processing**: Process multiple company reports simultaneously
- **AI-Powered Extraction**: Uses Google Gemini AI for accurate data extraction
- **Comprehensive KPIs**: Extract Scope 1, 2, 3 emissions, energy usage, and intensity metrics
- **Smart Page Ranking**: Intelligent algorithm finds the most relevant pages
- **Interactive Data Editor**: Easy-to-use table for inputting report details
- **Export to Excel**: Download results with clickable links to source pages
- **Visual Verification**: Preview the actual PDF pages where data was found

## 🚀 Quick Start

### Option 1: Use the Live App
Visit our deployed Streamlit app: [Climate Heroes KPI Extractor](https://your-app-url.streamlit.app)

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/climate-heroes-kpi-extractor.git
   cd climate-heroes-kpi-extractor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   - Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create `.streamlit/secrets.toml` and add:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## 📊 Supported KPIs

- **Scope 1 Emissions**: Direct GHG emissions
- **Scope 2 Emissions**: Indirect energy emissions (market-based and location-based)
- **Scope 3 Emissions**: Value chain emissions
- **Total Energy Usage**: Overall energy consumption
- **Emissions Intensity**: Carbon intensity metrics
- **Energy Intensity**: Energy efficiency metrics

## 🔧 Configuration

### Environment Variables
The app checks for API keys in this order:
1. `GEMINI_API_KEY_R5Y`
2. `GOOGLE_API_KEY_RK`
3. `GEMINI_API_KEY` (recommended)

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `GEMINI_API_KEY` in the app settings under "Secrets"

## 🛠️ Development

### Project Structure
```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .streamlit/
│   └── secrets.toml.example       # API key template
└── .gitignore                     # Git ignore file
```

### Key Components
- **Page Ranking Algorithm**: Scores PDF pages based on table presence, target years, and KPI keywords
- **AI Extraction**: Uses Gemini AI with temperature=0 for consistent results
- **Batch Processing**: Handles multiple reports with progress tracking
- **Error Handling**: Graceful fallbacks for different API versions

## 📝 Usage Examples

### Single Report
```
Company: Meta
Year: 2023
URL: https://sustainability.atmeta.com/resources/reports/2024-sustainability-report
```

### Multiple Reports
```
Meta, 2023, https://sustainability.atmeta.com/resources/reports/2024-sustainability-report
DBS, 2024, https://www.dbs.com/iwov-resources/images/sustainability/reporting/pdf/web/DBS_SR2024.pdf
Microsoft, 2023, https://aka.ms/2023ESGReport
```

## 🔍 Troubleshooting

**API Key Issues**
- Ensure your `.streamlit/secrets.toml` file exists and contains a valid API key
- Check that the key name matches one of the supported formats

**PDF Processing Issues**
- Verify URLs are publicly accessible and point directly to PDF files
- Some PDFs may be behind authentication - these won't work

**Installation Issues**
- On Windows, if PyMuPDF fails to install, update pip first: `python -m pip install --upgrade pip`
- For M1 Macs, you may need to install some dependencies via conda

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini AI](https://ai.google.dev/)
- PDF processing with [PyMuPDF](https://pymupdf.readthedocs.io/)

## 📧 Support


For questions or support, please open an issue on GitHub or contact [digitalsriart@gmail.com]
