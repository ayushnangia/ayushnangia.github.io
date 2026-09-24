const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');

const html = readFileSync('index.html', 'utf8');
const script = html.match(/<script>\s*([\s\S]*?const dicta =[\s\S]*?)<\/script>/)[1]
  .replace('{{ site.data.dicta | jsonify }}', JSON.stringify([
    { text: 'First' }, { text: 'Second', source: 'Author' },
  ]));
const elements = {
  '#dictum-text': { textContent: '' },
  '#dictum-source': { textContent: '', hidden: false },
};
const timers = [];
let clock = new Date(2026, 8, 24, 23, 59).getTime();
class TestDate extends Date {
  constructor(...args) { super(...(args.length ? args : [clock])); }
}
runInNewContext(script, {
  Date: TestDate,
  document: { querySelector: selector => elements[selector] },
  setTimeout: (callback, delay) => timers.push({ callback, delay }),
});
const first = elements['#dictum-text'].textContent;
assert.ok(timers[0].delay > 0 && timers[0].delay < 61000);
clock = new Date(2026, 8, 25, 0, 0).getTime();
timers[0].callback();
assert.notEqual(elements['#dictum-text'].textContent, first);
assert.equal(elements['#dictum-source'].hidden, first === 'Second');
console.log('Daily rotation and midnight update work');
