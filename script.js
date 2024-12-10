class HeaderBar extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
        <header>
        <div style="overflow:hidden">
          <span style="float:left">
            <a href="/">home</a>
          </span>
          <span style="float:right">
            <a href="https://github.com/stoksc">github</a>
            <a href="https://www.linkedin.com/in/bradleylaney/">linkedin</a>
          </span>
        </div>
      </header>
`;
  }
}

customElements.define("header-bar", HeaderBar);
