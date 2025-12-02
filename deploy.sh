#!/bin/bash

# Dental AI Deployment Script
# اسکریپت نصب و راه‌اندازی خودکار پروژه Dental AI

set -e  # Exit on error

echo "=========================================="
echo "Dental AI Deployment Script"
echo "اسکریپت نصب و راه‌اندازی Dental AI"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/var/www/dental-ai"
DOMAIN_NAME="${1:-your-domain.com}"
DB_NAME="dental_ai"
DB_USER="dental_ai_user"
DB_PASSWORD="${2:-$(openssl rand -base64 32)}"
JWT_SECRET="${3:-$(openssl rand -base64 32)}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project Directory: $PROJECT_DIR"
echo "  Domain: $DOMAIN_NAME"
echo "  Database Name: $DB_NAME"
echo "  Database User: $DB_USER"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Step 1: Update system
echo -e "${GREEN}[1/10] Updating system...${NC}"
apt-get update && apt-get upgrade -y

# Step 2: Install prerequisites
echo -e "${GREEN}[2/10] Installing prerequisites...${NC}"
apt-get install -y \
    git curl wget build-essential \
    nginx \
    postgresql postgresql-contrib \
    python3.8 python3.8-venv python3-pip python3-dev \
    certbot python3-certbot-nginx \
    ufw

# Step 3: Install Node.js
echo -e "${GREEN}[3/10] Installing Node.js...${NC}"
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
fi

# Step 4: Install PM2
echo -e "${GREEN}[4/10] Installing PM2...${NC}"
npm install -g pm2

# Step 5: Create project directory
echo -e "${GREEN}[5/10] Creating project directory...${NC}"
mkdir -p $PROJECT_DIR
mkdir -p $PROJECT_DIR/uploads
mkdir -p /var/backups/dental-ai
mkdir -p /var/log/dental-ai

# Step 6: Setup database
echo -e "${GREEN}[6/10] Setting up database...${NC}"
sudo -u postgres psql <<EOF
CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
\q
EOF

# Step 7: Setup Python virtual environment
echo -e "${GREEN}[7/10] Setting up Python environment...${NC}"
cd $PROJECT_DIR
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Note: User needs to install Python dependencies manually
# pip install -r requirements_unified_api.txt

# Step 8: Setup Frontend
echo -e "${GREEN}[8/10] Setting up Frontend...${NC}"
if [ -d "$PROJECT_DIR/vite-js" ]; then
    cd $PROJECT_DIR/vite-js
    npm install
    npm run build
else
    echo -e "${YELLOW}Warning: vite-js directory not found. Please build manually.${NC}"
fi

# Step 9: Setup Backend
echo -e "${GREEN}[9/10] Setting up Backend...${NC}"
if [ -d "$PROJECT_DIR/minimal-api-dev-v6" ]; then
    cd $PROJECT_DIR/minimal-api-dev-v6
    npm install
    
    # Create .env.production
    cat > .env.production <<ENVEOF
NODE_ENV=production
PORT=7272
HOST=0.0.0.0
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME
JWT_SECRET=$JWT_SECRET
AI_SERVER_URL=http://localhost:5001
MAX_FILE_SIZE=10485760
UPLOAD_DIR=$PROJECT_DIR/uploads
ALLOWED_ORIGINS=https://$DOMAIN_NAME,https://www.$DOMAIN_NAME
ENVEOF
    
    npm run build
    
    # Setup PM2
    cat > ecosystem.config.js <<PM2EOF
module.exports = {
  apps: [{
    name: 'dental-ai-api',
    script: 'npm',
    args: 'start',
    cwd: '$PROJECT_DIR/minimal-api-dev-v6',
    env: {
      NODE_ENV: 'production',
      PORT: 7272
    },
    error_file: '/var/log/dental-ai/api-error.log',
    out_file: '/var/log/dental-ai/api-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
};
PM2EOF
    
    pm2 start ecosystem.config.js
    pm2 save
    pm2 startup
else
    echo -e "${YELLOW}Warning: minimal-api-dev-v6 directory not found.${NC}"
fi

# Step 10: Setup Python AI Server systemd service
echo -e "${GREEN}[10/10] Setting up Python AI Server...${NC}"
cat > /etc/systemd/system/dental-ai-python.service <<SERVICEEOF
[Unit]
Description=Dental AI Python Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/.venv/bin"
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/unified_ai_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable dental-ai-python
# Note: Start after installing Python dependencies
# systemctl start dental-ai-python

# Step 11: Setup Nginx
echo -e "${GREEN}[11/10] Setting up Nginx...${NC}"
# Frontend config
cat > /etc/nginx/sites-available/dental-ai-frontend <<NGINXEOF
server {
    listen 80;
    server_name $DOMAIN_NAME www.$DOMAIN_NAME;

    root $PROJECT_DIR/vite-js/dist;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)\$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json;
}
NGINXEOF

# Backend API config
cat > /etc/nginx/sites-available/dental-ai-api <<NGINXEOF
server {
    listen 80;
    server_name api.$DOMAIN_NAME;

    location / {
        proxy_pass http://localhost:7272;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
NGINXEOF

ln -sf /etc/nginx/sites-available/dental-ai-frontend /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/dental-ai-api /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Step 12: Setup Firewall
echo -e "${GREEN}[12/10] Setting up Firewall...${NC}"
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp

# Step 13: Setup SSL (optional - requires domain)
echo -e "${YELLOW}Note: SSL certificate setup requires domain to be pointed to this server.${NC}"
echo -e "${YELLOW}Run the following command after DNS is configured:${NC}"
echo "  sudo certbot --nginx -d $DOMAIN_NAME -d www.$DOMAIN_NAME -d api.$DOMAIN_NAME"

# Step 14: Setup Backup Script
echo -e "${GREEN}[13/10] Setting up Backup Script...${NC}"
cat > /usr/local/bin/backup-dental-ai-db.sh <<BACKUPEOF
#!/bin/bash
BACKUP_DIR="/var/backups/dental-ai"
DATE=\$(date +%Y%m%d_%H%M%S)
mkdir -p \$BACKUP_DIR
PGPASSWORD='$DB_PASSWORD' pg_dump -U $DB_USER $DB_NAME > \$BACKUP_DIR/db_backup_\$DATE.sql
find \$BACKUP_DIR -name "*.sql" -mtime +7 -delete
BACKUPEOF

chmod +x /usr/local/bin/backup-dental-ai-db.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-dental-ai-db.sh") | crontab -

# Set permissions
chown -R www-data:www-data $PROJECT_DIR
chmod -R 755 $PROJECT_DIR
chmod -R 775 $PROJECT_DIR/uploads

echo ""
echo -e "${GREEN}=========================================="
echo "Deployment Complete!"
echo "نصب و راه‌اندازی کامل شد!"
echo "==========================================${NC}"
echo ""
echo -e "${YELLOW}Important Information:${NC}"
echo "  Database Name: $DB_NAME"
echo "  Database User: $DB_USER"
echo "  Database Password: $DB_PASSWORD"
echo "  JWT Secret: $JWT_SECRET"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Install Python dependencies:"
echo "     cd $PROJECT_DIR"
echo "     source .venv/bin/activate"
echo "     pip install -r requirements_unified_api.txt"
echo "     pip install face-alignment mediapipe scikit-image"
echo ""
echo "  2. Run database migrations:"
echo "     cd $PROJECT_DIR/minimal-api-dev-v6"
echo "     npx prisma migrate deploy"
echo ""
echo "  3. Start Python AI Server:"
echo "     sudo systemctl start dental-ai-python"
echo ""
echo "  4. Setup SSL (after DNS is configured):"
echo "     sudo certbot --nginx -d $DOMAIN_NAME -d www.$DOMAIN_NAME -d api.$DOMAIN_NAME"
echo ""
echo -e "${GREEN}Save the credentials above in a secure location!${NC}"

