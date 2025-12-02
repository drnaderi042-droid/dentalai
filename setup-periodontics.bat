@echo off
REM Setup Periodontics System
REM This script sets up the database and dependencies for the Periodontics module

echo 🦷 Setting up Periodontics System...

REM Navigate to backend
cd minimal-api-dev-v6

echo 📦 Installing dependencies...
call npm install

echo 🗄️ Generating Prisma Client...
call npx prisma generate

echo 📊 Creating database migration...
call npx prisma migrate dev --name add_periodontal_charts

echo ✅ Backend setup complete!

REM Navigate to frontend
cd ..\vite-js

echo 📦 Installing frontend dependencies...
call npm install

echo ✅ Frontend setup complete!

echo.
echo 🎉 Periodontics System is ready!
echo.
echo To start the servers:
echo   Backend:  cd minimal-api-dev-v6 ^&^& npm run dev
echo   Frontend: cd vite-js ^&^& npm run dev
echo.
echo Then visit: http://localhost:3031/dashboard/periodontics

pause



