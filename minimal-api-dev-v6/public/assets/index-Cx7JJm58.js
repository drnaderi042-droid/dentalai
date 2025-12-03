import{r as i,j as e,p as m,bf as l,S as p,T as c,H as d,C as u}from"./index-BSItzkt3.js";import{E as h}from"./editor-cOR4tKCy.js";import{M as x}from"./markdown-DkFU1EoS.js";import{C as f}from"./custom-breadcrumbs-u_kfWYpE.js";import{C}from"./component-hero-McIacDkb.js";import{C as g}from"./component-block-DUg0sejQ.js";import{C as j}from"./Card-BqWDE3fr.js";import{F as k}from"./FormControlLabel-CtXEP1QR.js";import"./TextField-DpdcsLRb.js";import"./Select-QPCMlUZ4.js";import"./Menu-Cy0n4kRw.js";import"./InputLabel-DqZ4yv4U.js";import"./FormLabel-Dg62SF9b.js";import"./FormHelperText-CBXzj8d7.js";import"./FormControl-1z33WGqh.js";import"./index-DC_o48Ax.js";import"./html-to-markdown-CuRjBHl-.js";import"./image-BuqYs7tq.js";const b=`

<h4>This is Heading 4</h4>
<code>This is code</code>

<pre><code class="language-javascript">for (var i=1; i &#x3C;= 20; i++) {
  if (i % 15 == 0)
    return "FizzBuzz"
  else if (i % 3 == 0)
    return "Fizz"
  else if (i % 5 == 0)
    return "Buzz"
  else
    return i
  }</code></pre>
`;function z(){const[r,s]=i.useState(!0),[o,a]=i.useState(b),n=t=>{s(t.target.checked)};return e.jsxs(e.Fragment,{children:[e.jsx(C,{children:e.jsx(f,{heading:"Editor",links:[{name:"Components",href:m.components},{name:"Editor"}],moreLink:["https://tiptap.dev/docs/editor/introduction"]})}),e.jsxs(g,{maxWidth:!1,sx:{rowGap:5,columnGap:3,display:"grid",gridTemplateColumns:{xs:"repeat(1, 1fr)",md:"repeat(2, 1fr)"}},children:[e.jsxs(j,{sx:{p:3,gap:2,flexShrink:0,display:"flex",flexDirection:"column"},children:[e.jsx(k,{control:e.jsx(l,{name:"fullItem",checked:r,onChange:n}),label:"Full item",labelPlacement:"start",sx:{ml:"auto"}}),e.jsx(h,{fullItem:r,value:o,onChange:t=>a(t),sx:{maxHeight:720}})]}),e.jsxs(p,{spacing:1,sx:{p:3,borderRadius:2,overflowX:"auto",bgcolor:"background.neutral"},children:[e.jsx(c,{variant:"h6",children:"Preview"}),e.jsx(x,{children:o})]})]})]})}const E={title:`Editor | Components - ${u.site.name}`};function W(){return e.jsxs(e.Fragment,{children:[e.jsx(d,{children:e.jsxs("title",{children:[" ",E.title]})}),e.jsx(z,{})]})}export{W as default};
