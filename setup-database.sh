#!/bin/bash

# DentalAI - Database Setup Script
# این اسکریپت PostgreSQL را نصب و تنظیم می‌کند

set -e

echo "🐘 DentalAI - PostgreSQL Database Setup"
echo "========================================="

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

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    print_info "Installing PostgreSQL..."
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
    print_status "PostgreSQL installed"
else
    print_status "PostgreSQL is already installed"
fi

# Start PostgreSQL service
print_info "Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql
print_status "PostgreSQL service started"

# Generate random password
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
DB_USER="dentalai_user"
DB_NAME="dentalai"

print_info "Database configuration:"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo "  Password: $DB_PASSWORD"
echo ""

# Create database and user
print_info "Creating database and user..."

sudo -u postgres psql <<EOF
-- Create database
CREATE DATABASE $DB_NAME;

-- Create user
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- For PostgreSQL 15+
ALTER DATABASE $DB_NAME OWNER TO $DB_USER;

-- Exit
\q
EOF

print_status "Database and user created"

# Update pg_hba.conf for local connections
print_info "Configuring PostgreSQL authentication..."

PG_VERSION=$(psql --version | grep -oP '\d+' | head -1)
PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"

if [ -f "$PG_HBA" ]; then
    # Backup original file
    sudo cp "$PG_HBA" "$PG_HBA.backup"
    
    # Add local connection rules if not exists
    if ! grep -q "dentalai" "$PG_HBA"; then
        echo "" | sudo tee -a "$PG_HBA"
        echo "# DentalAI local connections" | sudo tee -a "$PG_HBA"
        echo "local   $DB_NAME        $DB_USER                           md5" | sudo tee -a "$PG_HBA"
        echo "host    $DB_NAME        $DB_USER   127.0.0.1/32            md5" | sudo tee -a "$PG_HBA"
        echo "host    $DB_NAME        $DB_USER   ::1/128                 md5" | sudo tee -a "$PG_HBA"
        
        print_status "Authentication configured"
    else
        print_warning "Authentication already configured"
    fi
    
    # Restart PostgreSQL
    sudo systemctl restart postgresql
    print_status "PostgreSQL restarted"
else
    print_warning "Could not find pg_hba.conf, manual configuration may be needed"
fi

# Test connection
print_info "Testing database connection..."
if PGPASSWORD="$DB_PASSWORD" psql -h localhost -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
    print_status "Database connection successful!"
else
    print_warning "Connection test failed, but database is created"
    print_info "You may need to configure authentication manually"
fi

# Create .env file
print_info "Creating .env file..."

ENV_FILE="/home/salahk/.env"
if [ -f "$ENV_FILE" ]; then
    print_warning ".env file already exists, backing up..."
    cp "$ENV_FILE" "$ENV_FILE.backup"
fi

# Generate secrets
NEXTAUTH_SECRET=$(openssl rand -base64 32 | tr -d "=+/")
JWT_SECRET=$(openssl rand -base64 32 | tr -d "=+/")

cat > "$ENV_FILE" <<EOF
# ============================================================================
# DentalAI - Environment Configuration
# Generated automatically by setup-database.sh
# ============================================================================

# Database Configuration
DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME?schema=public"

# Authentication & Security
NEXTAUTH_SECRET="$NEXTAUTH_SECRET"
NEXTAUTH_URL="https://ceph.bioritalin.ir"
JWT_SECRET="$JWT_SECRET"
BCRYPT_ROUNDS=12

# API URLs
VITE_API_URL="https://ceph.bioritalin.ir"
VITE_AI_API_URL="https://ceph.bioritalin.ir"
NEXT_PUBLIC_API_URL="https://ceph.bioritalin.ir"

# Application Settings
NODE_ENV="production"
FLASK_ENV="production"
PORT=7272

# Python AI Server Settings
PYTHONPATH="/home/salahk"
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
VECLIB_MAXIMUM_THREADS=2
NUMEXPR_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=""

# File Upload Settings
UPLOAD_DIR="/home/salahk/uploads"
MAX_FILE_SIZE=104857600

# Logging
LOG_LEVEL="INFO"
LOG_DIR="/home/salahk/logs"
EOF

print_status ".env file created at $ENV_FILE"

# Display summary
echo ""
echo "========================================="
echo "✅ Database Setup Completed!"
echo "========================================="
echo ""
echo "📋 Database Information:"
echo "   Database: $DB_NAME"
echo "   Username: $DB_USER"
echo "   Password: $DB_PASSWORD"
echo ""
echo "📝 Next Steps:"
echo "   1. Update Prisma schema for PostgreSQL:"
echo "      cd /home/salahk/backend"
echo "      nano prisma/schema.prisma"
echo "      # Change: provider = \"postgresql\""
echo ""
echo "   2. Generate Prisma Client:"
echo "      npx prisma generate"
echo ""
echo "   3. Run migrations:"
echo "      npx prisma db push"
echo ""
echo "   4. Verify database:"
echo "      npx prisma studio"
echo ""
echo "🔐 Security Note:"
echo "   - Database password saved in: $ENV_FILE"
echo "   - Keep this file secure!"
echo "   - Backup file created at: $ENV_FILE.backup"
echo ""
print_status "Setup completed successfully! 🎉"



