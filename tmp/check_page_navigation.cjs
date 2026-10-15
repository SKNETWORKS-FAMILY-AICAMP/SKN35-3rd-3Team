const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/app/page_navigation.js', 'utf8');
const doc = new EventTarget();
const win = new EventTarget();
let focused = false;
let reduced = false;
let observer;
let button = new EventTarget();
button.hidden = true;
const main = {scrollTop: 0, calls: [], scrollTo(options) {this.calls.push(options); this.scrollTop = options.top;}};
doc.scrollingElement = {scrollTop: 0, scrollTo(options) {this.scrollTop = options.top;}};
doc.body = {};
doc.getElementById = () => button;
doc.querySelector = selector => selector.includes('stMain') ? main : {focus() {focused = true;}};
win.scrollY = 0;
win.scrollTo = options => {win.scrollY = options.top; doc.dispatchEvent(new Event('scroll'));};
win.matchMedia = () => ({matches: reduced});
class Observer {
  constructor(callback) {this.callback = callback; observer = this;}
  observe() {}
  disconnect() {this.disconnected = true;}
}
const context = vm.createContext({window: win, document: doc, MutationObserver: Observer});
vm.runInContext(source, context);
assert.equal(button.hidden, true);
// A nested chat scrolling does not change the page offset.
doc.dispatchEvent(new Event('scroll'));
assert.equal(button.hidden, true);
main.scrollTop = 400;
doc.dispatchEvent(new Event('scroll'));
assert.equal(button.hidden, false);
button.dispatchEvent(new Event('click'));
assert.equal(main.calls.length, 1);
assert.equal(main.calls[0].behavior, 'smooth');
assert.equal(button.hidden, true);
assert.equal(focused, true);
// Rerendering replaces handlers instead of registering duplicates.
vm.runInContext(source, context);
reduced = true;
button.dispatchEvent(new Event('click'));
assert.equal(main.calls.length, 2);
assert.equal(main.calls[1].behavior, 'auto');
// React may mount the button after the script; the observer handles this.
button = null;
vm.runInContext(source, context);
button = new EventTarget();
observer.callback();
assert.equal(observer.disconnected, true);
assert.equal(button.hidden, true);
win.__lawMatePageNavigation.cleanup();
main.scrollTop = 500;
doc.dispatchEvent(new Event('scroll'));
assert.equal(button.hidden, true);
console.log('Top visibility, page scroll, reduced motion, rerender and cleanup OK');
