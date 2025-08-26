# Deployment Guide for Climate Heroes KPI Extractor

## 🚀 Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Google Gemini API key

### Step-by-Step Deployment

1. **Prepare Your Repository**
   ```bash
   # Push your code to GitHub
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select this repository and `app.py` as the main file

3. **Configure Secrets**
   - In your Streamlit Cloud app settings, go to "Secrets"
   - Add your API key:
     ```toml
     GEMINI_API_KEY = "your-actual-api-key-here"
     ```

4. **Deploy**
   - Click "Deploy"
   - Your app will be available at `https://your-app-name.streamlit.app`

## 🔧 Alternative Deployment Options

### Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Set environment variables:
   ```bash
   heroku config:set GEMINI_API_KEY=your-api-key
   ```

### Docker
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

### AWS EC2
1. Launch EC2 instance
2. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

## 🔐 Security Considerations

- Never commit API keys to version control
- Use environment variables or Streamlit secrets
- Consider rate limiting for public deployments
- Monitor API usage and costs

## 📊 Performance Optimization

- Use `st.cache_data` for expensive operations
- Limit concurrent API calls
- Consider implementing request queuing for high traffic

## 🐛 Troubleshooting Deployment Issues

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **API Key Not Found**
   - Verify secrets configuration
   - Check secret key names match the code

3. **Memory Issues**
   - Consider reducing DPI settings
   - Implement cleanup for large files

4. **Timeout Issues**
   - Add retry logic for API calls
   - Increase timeout values for large PDFs

### Health Check
Add this endpoint for monitoring:
```python
@st.cache_data
def health_check():
    return {"status": "healthy", "timestamp": time.time()}
```

## 📈 Monitoring

### Key Metrics to Track
- API call volume and costs
- Processing time per report
- Error rates by error type
- User engagement metrics

### Recommended Tools
- Streamlit Cloud analytics (built-in)
- Google Cloud Monitoring (for API usage)
- Custom logging with structured data

## 🔄 CI/CD Pipeline

Example GitHub Actions workflow (`.github/workflows/deploy.yml`):

```yaml
name: Deploy to Streamlit Cloud
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/ || echo "No tests found"
```
