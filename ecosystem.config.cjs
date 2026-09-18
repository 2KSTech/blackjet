// pm2 process file for blackjet. Place in the project root, beside run.sh.
// run.sh sources .env and execs the server, so no env block is needed here.
//   pm2 start ecosystem.config.cjs
//   pm2 save && pm2 startup     # survive reboots
module.exports = {
  apps: [
    {
      name: "blackjet",
      script: "./run.sh",
      interpreter: "bash",
      cwd: __dirname,
      autorestart: true,
      restart_delay: 5000,
      kill_timeout: 10000,
      time: true,
    },
  ],
};

