#!/usr/bin/env node
/**
 * serve.js – Minimal static file server for the MambaCode.js tools.
 *
 * Serves the project root over HTTP so that browser ES module imports work.
 * A local server is required because browsers block ES module imports from
 * file:// URLs due to CORS restrictions.
 *
 * Usage:
 *   node serve.js           # serves on http://localhost:3000
 *   node serve.js 8080      # custom port
 */

import http from 'http';
import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT      = parseInt(process.argv[2], 10) || 3000;
const ROOT      = __dirname;

const MIME = {
  '.html' : 'text/html; charset=utf-8',
  '.js'   : 'application/javascript; charset=utf-8',
  '.mjs'  : 'application/javascript; charset=utf-8',
  '.ts'   : 'application/javascript; charset=utf-8',
  '.json' : 'application/json; charset=utf-8',
  '.css'  : 'text/css; charset=utf-8',
  '.map'  : 'application/json; charset=utf-8',
  '.bin'  : 'application/octet-stream',
  '.txt'  : 'text/plain; charset=utf-8',
  '.ico'  : 'image/x-icon',
};

const server = http.createServer((req, res) => {
  // Strip query string and decode
  const urlPath  = decodeURIComponent(req.url.split('?')[0]);
  const filePath = path.join(ROOT, urlPath === '/' ? '/tools/pretrain.html' : urlPath);

  // Guard against path traversal
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`Not found: ${urlPath}`);
      return;
    }

    const ext         = path.extname(filePath).toLowerCase();
    const contentType = MIME[ext] ?? 'application/octet-stream';

    res.writeHead(200, { 'Content-Type': contentType, 'Cache-Control': 'no-cache' });
    fs.createReadStream(filePath).pipe(res);
  });
});

function printLinks(port) {
  console.log(`\nMambaCode.js dev server running at http://localhost:${port}\n`);
  console.log('  Tools:');
  console.log('  → http://localhost:' + port + '/tools/pretrain.html   (browser pretraining)');
  console.log('  → http://localhost:' + port + '/tools/convert.html    (HuggingFace → MBJS converter)');
  console.log('\nPress Ctrl+C to stop.\n');
}

server.once('listening', () => printLinks(server.address().port));

(function tryListen(port) {
  server.once('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.warn(`Port ${port} in use, trying ${port + 1}…`);
      tryListen(port + 1);
    } else {
      throw err;
    }
  });
  server.listen(port);
}(PORT));
