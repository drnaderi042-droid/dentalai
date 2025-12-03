import{r as i,j as e,p as m,bf as l,S as p,T as c,H as d,C as u}from"./index-BH9ijvcg.js";import{E as h}from"./editor-BUp8U2Vl.js";import{M as x}from"./markdown-DRxN9prE.js";import{C as f}from"./custom-breadcrumbs-C3JvxER0.js";import{C}from"./component-hero-Biw0R1u6.js";import{C as g}from"./component-block-C7-s7KXv.js";import{C as j}from"./Card-CwEGl0A2.js";import{F as k}from"./FormControlLabel-CVzZoZzb.js";import"./TextField-DgF_8CdY.js";import"./Select-fMsy-Lkz.js";import"./Menu-C7S9TApF.js";import"./InputLabel-OzeQogzC.js";import"./FormLabel-Cn8enUsF.js";import"./FormHelperText-CAGzEAMm.js";import"./FormControl-rHOkSiGz.js";import"./index-BVzUWzoE.js";import"./html-to-markdown-Daz6g1i5.js";import"./image-CxkC95SJ.js";const b=`

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
