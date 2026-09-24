const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');

const html = readFileSync('index.html', 'utf8');
const script = html.match(/<script>\s*(const modeButton =[\s\S]*?)<\/script>/)[1];
const button = {
  textContent: 'Silly mode',
  addEventListener: (_event, callback) => { button.click = callback; },
};
const portrait = { src: 'images/ayush-serious.jpg', alt: 'Portrait of Ayush Nangia' };
runInNewContext(script, { document: { querySelector: selector => selector === '#mode-toggle' ? button : portrait } });

button.click();
assert.equal(portrait.src, '/images/ayush-llama.jpg');
assert.equal(portrait.alt, 'Ayush Nangia standing beside a llama');
assert.equal(button.textContent, 'Serious mode');
button.click();
assert.equal(portrait.src, '/images/ayush-serious.jpg');
assert.equal(portrait.alt, 'Portrait of Ayush Nangia');
assert.equal(button.textContent, 'Silly mode');
console.log('Photo mode toggle works');
