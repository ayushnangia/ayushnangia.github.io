const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');

const html = readFileSync('index.html', 'utf8');
const script = html.match(/<script>\s*(const modeButton =[\s\S]*?)<\/script>/)[1];
const button = {
  setAttribute: (name, value) => { button[name] = value; },
  addEventListener: (_event, callback) => { button.click = callback; },
};
const portrait = { src: 'images/ayush-serious.jpg', alt: 'Portrait of Ayush Nangia' };
const mouth = { setAttribute: (_name, value) => { mouth.path = value; } };
runInNewContext(script, { document: { querySelector: selector => ({ '#mode-toggle': button, '#portrait': portrait, '#mode-mouth': mouth })[selector] } });

button.click();
assert.equal(portrait.src, '/images/ayush-llama.jpg');
assert.equal(portrait.alt, 'Ayush Nangia standing beside a llama');
assert.equal(button['aria-label'], 'Show serious photo');
assert.equal(mouth.path, 'M8 15h8');
button.click();
assert.equal(portrait.src, '/images/ayush-serious.jpg');
assert.equal(portrait.alt, 'Portrait of Ayush Nangia');
assert.equal(button['aria-label'], 'Show llama photo');
assert.equal(mouth.path, 'M8 14c1.5 2 6.5 2 8 0');
console.log('Photo mode toggle works');
