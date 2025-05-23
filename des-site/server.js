import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname, '')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

import ImageProcessor from './processor.js';

console.log('ImageProcessor has been imported:', ImageProcessor);

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
