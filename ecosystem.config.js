module.exports = {
  apps: [
    {
      name: 'nextjs-api',
      script: './server.js',
      cwd: './minimal-api-dev-v6',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: 'development',
        PORT: 443
      },
      env_production: {
        NODE_ENV: 'development',
        PORT: 443
      },
      error_file: '../logs/nextjs-api-error.log',
      out_file: '../logs/nextjs-api-out.log',
      log_file: '../logs/nextjs-api.log',
      merge_logs: true,
      time: true
    },
    {
      name: 'python-ai-api',
      script: './unified_ai_api_server.py',
      interpreter: 'python3',
      instances: 1,
      exec_mode: 'fork',
      env: {
        FLASK_ENV: 'production',
        PORT: 5001
      },
      env_production: {
        FLASK_ENV: 'production',
        PORT: 5001
      },
      error_file: './logs/python-ai-api-error.log',
      out_file: './logs/python-ai-api-out.log',
      log_file: './logs/python-ai-api.log',
      merge_logs: true,
      time: true
    }
  ]
};
