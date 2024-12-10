class HeaderBar extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <header>
        <a href="/">home</a>
        <span style="float:right;">
          <a href="https://github.com/stoksc">github</a>
          <a href="https://www.linkedin.com/in/bradleylaney/">linkedin</a>
        </span>
      </header>
`;
  }
}

customElements.define("header-bar", HeaderBar);
