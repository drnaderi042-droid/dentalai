#!/bin/bash

# DentalAI - OLSPanel Setup Script
# این اسکریپت برای راه‌اندازی پروژه DentalAI روی OLSPanel آماده شده

set -e

echo "🐼 DentalAI - OLSPanel Deployment Setup"
echo "======================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're running on a server with OLSPanel
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Make sure you're running this on a server with OLSPanel installed."
    print_info "OLSPanel includes Node.js. If not installed, visit: https://olspanel.com/"
    exit 1
fi

print_status "Node.js version: $(node --version)"
print_status "npm version: $(npm --version)"

# Check if Python is available
if command -v python3 &> /dev/null; then
    print_status "Python3 found: $(python3 --version)"
else
    print_warning "Python3 not found. Installing..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
fi

# Check if PostgreSQL is available
if command -v psql &> /dev/null; then
    print_status "PostgreSQL found"
else
    print_warning "PostgreSQL not found. Installing..."
    sudo apt install -y postgresql postgresql-contrib
fi

print_info "Setting up DentalAI project structure..."

# Create project directory if it doesn't exist
PROJECT_DIR="$HOME/dentalai"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
    print_status "Created project directory: $PROJECT_DIR"
else
    print_warning "Project directory already exists: $PROJECT_DIR"
fi

# Setup database
print_info "Setting up PostgreSQL database..."
sudo -u postgres psql -c "CREATE DATABASE dentalai;" 2>/dev/null || print_warning "Database 'dentalai' might already exist"
sudo -u postgres psql -c "CREATE USER dentalai_user WITH PASSWORD 'dentalai2024';" 2>/dev/null || print_warning "User 'dentalai_user' might already exist"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE dentalai TO dentalai_user;" 2>/dev/null || true

print_status "Database setup completed"

# Create environment file
print_info "Creating environment configuration..."
cat > "$PROJECT_DIR/.env" << EOF
# DentalAI Environment Configuration for OLSPanel

# Database
DATABASE_URL="postgresql://dentalai_user:dentalai2024@localhost:5432/dentalai"

# Authentication
NEXTAUTH_SECRET=dentalai-olspanel-secret-key-$(date +%s)
NEXTAUTH_URL=http://localhost:7272

# API URLs
VITE_API_URL=http://localhost:7272
VITE_AI_API_URL=http://localhost:5000
NEXT_PUBLIC_API_URL=http://localhost:7272

# Application Settings
NODE_ENV=production
FLASK_ENV=production
PYTHONPATH=$PROJECT_DIR

# AI/ML Settings
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=""

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=104857600

# Security
JWT_SECRET=dentalai-jwt-secret-$(date +%s)
BCRYPT_ROUNDS=12

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
EOF

print_status "Environment file created: $PROJECT_DIR/.env"

# Create deployment instructions
print_info "Creating deployment instructions..."
cat > "$PROJECT_DIR/OLSPANEL_DEPLOYMENT_STEPS.md" << 'EOF'
# DentalAI - OLSPanel Deployment Steps

## مرحله ۱: انتقال فایل‌ها
```
# فایل‌های پروژه را در دایرکتوری زیر قرار دهید:
# /home/your-user/dentalai/

# ساختار نهایی:
# dentalai/
# ├── vite-js/          # Frontend (پورت 3030)
# ├── minimal-api-dev-v6/  # Backend API (پورت 7272)
# ├── next-js/          # Dashboard (پورت 3000)
# ├── unified_ai_api_server.py  # AI Server (پورت 5000)
# └── requirements_unified_api.txt
```

## مرحله ۲: تنظیم Sites در OLSPanel

### ۲.۱ Frontend (vite-js)
1. **Sites** → **Add Site**
2. **Domain**: `dentalai.yourdomain.com`
3. **Application Type**: `Node.js`
4. **Node.js Version**: `18`
5. **Application Path**: `/home/user/dentalai/vite-js`
6. **Start Command**: `npm run dev -- --host 0.0.0.0 --port 3030`
7. **Port**: `3030`

**Environment Variables:**
```
NODE_ENV=production
VITE_API_URL=http://localhost:7272
VITE_AI_API_URL=http://localhost:5000
```

### ۲.۲ Backend API (minimal-api-dev-v6)
1. **Sites** → **Add Site**
2. **Domain**: `api.dentalai.yourdomain.com`
3. **Application Type**: `Node.js`
4. **Node.js Version**: `18`
5. **Application Path**: `/home/user/dentalai/minimal-api-dev-v6`
6. **Start Command**: `npm run dev -- -p 7272`
7. **Port**: `7272`

**Environment Variables:**
```
DATABASE_URL=postgresql://dentalai_user:dentalai2024@localhost:5432/dentalai
NEXTAUTH_SECRET=your-secret-key
NEXTAUTH_URL=http://api.dentalai.yourdomain.com
NODE_ENV=production
```

### ۲.۳ AI Server (Python)
1. **Sites** → **Add Site**
2. **Domain**: `ai.dentalai.yourdomain.com`
3. **Application Type**: `Python`
4. **Python Version**: `3.x`
5. **Application Path**: `/home/user/dentalai`
6. **Start Command**: `python unified_ai_api_server.py`
7. **Port**: `5000`

### ۲.۴ Dashboard (next-js)
1. **Sites** → **Add Site**
2. **Domain**: `dashboard.dentalai.yourdomain.com`
3. **Application Type**: `Node.js`
4. **Node.js Version**: `18`
5. **Application Path**: `/home/user/dentalai/next-js`
6. **Start Command**: `npm run start`
7. **Port**: `3000`

## مرحله ۳: نصب وابستگی‌ها

### روی سرور (SSH):
```bash
cd ~/dentalai

# Frontend
cd vite-js && npm install

# Backend API
cd ../minimal-api-dev-v6 && npm install && npx prisma generate && npx prisma db push

# Dashboard
cd ../next-js && npm install && npm run build

# AI Server
cd .. && python3 -m venv dentalai-env && source dentalai-env/bin/activate
pip install -r requirements_unified_api.txt
```

## مرحله ۴: تنظیم SSL

1. برای هر Site: **SSL** → **Free SSL**
2. دامنه را وارد کنید
3. **Install SSL** کلیک کنید

## مرحله ۵: تنظیم Firewall

در OLSPanel:
1. **Security** → **Firewall**
2. پورت‌های زیر را باز کنید:
   - 3030 (Frontend)
   - 7272 (API)
   - 5000 (AI Server)
   - 3000 (Dashboard)

## دسترسی به برنامه:

- **Frontend**: https://dentalai.yourdomain.com
- **API**: https://api.dentalai.yourdomain.com
- **AI Server**: https://ai.dentalai.yourdomain.com
- **Dashboard**: https://dashboard.dentalai.yourdomain.com

## عیب‌یابی:

### بررسی وضعیت سرویس‌ها:
```bash
# بررسی پورت‌ها
netstat -tlnp | grep -E ':(3030|7272|5000|3000)'

# بررسی لاگ‌ها در OLSPanel
# Sites → انتخاب site → Logs
```

### راه‌اندازی مجدد:
اگر مشکلی پیش آمد، از طریق OLSPanel هر site را restart کنید.
EOF

print_status "Deployment instructions created: $PROJECT_DIR/OLSPANEL_DEPLOYMENT_STEPS.md"

# Create a simple health check script
print_info "Creating health check script..."
cat > "$PROJECT_DIR/health-check.sh" << 'EOF'
#!/bin/bash

echo "🔍 DentalAI Health Check"
echo "========================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    local port=$1
    local name=$2

    if curl -s http://localhost:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name (port $port): Running${NC}"
    else
        echo -e "${RED}❌ $name (port $port): Not responding${NC}"
    fi
}

# Check all services
check_service 3030 "Frontend (Vite)"
check_service 7272 "Backend API (Next.js)"
check_service 5000 "AI Server (Python)"
check_service 3000 "Dashboard (Next.js)"

# Check database
if psql -h localhost -U dentalai_user -d dentalai -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Database: Connected${NC}"
else
    echo -e "${RED}❌ Database: Connection failed${NC}"
fi

echo ""
echo "📋 Useful commands:"
echo "  - View OLSPanel logs: Check OLSPanel dashboard"
echo "  - Restart services: Use OLSPanel interface"
echo "  - Check resources: htop or top"
EOF

chmod +x "$PROJECT_DIR/health-check.sh"
print_status "Health check script created: $PROJECT_DIR/health-check.sh"

print_header "🎉 Setup completed!"
echo ""
print_info "Project directory: $PROJECT_DIR"
print_info "Environment file: $PROJECT_DIR/.env"
print_info "Instructions: $PROJECT_DIR/OLSPANEL_DEPLOYMENT_STEPS.md"
print_info "Health check: $PROJECT_DIR/health-check.sh"
echo ""
print_info "Next steps:"
echo "1. Upload your project files to: $PROJECT_DIR"
echo "2. Follow the instructions in: OLSPANEL_DEPLOYMENT_STEPS.md"
echo "3. Access OLSPanel at: http://your-server-ip:port"
echo ""
print_success "Happy DentalAI deployment! 🦷🤖"



