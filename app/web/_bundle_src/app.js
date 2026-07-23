
class Component extends DCLogic {
  constructor(props){
    super(props);
    this.LINES=[['Lyα',0.12157,0.7],['[OII]',0.37274,0.55],['Hγ',0.43405,0.16],['Hβ',0.48613,0.34],['[OIII]',0.50082,1.0],['Hα',0.65628,1.0],['[SII]',0.67164,0.14],['[SIII]',0.95311,0.12],['HeI',1.0830,0.18],['Paβ',1.2822,0.10],['Paα',1.8756,0.16]];
    this.BANDS=[['F606W',0.606],['F814W',0.814],['F115W',1.154],['F150W',1.501],['F200W',1.990],['F277W',2.786],['F356W',3.563],['F444W',4.421]];
    this.objs=this.makeObjects();
    let dec={}; try{dec=JSON.parse(localStorage.getItem('djaqc-decisions')||'{}')}catch(e){}
    this.state={selId:this.objs[0].dja,wmin:0.6,wmax:5.3,mm:'both',zSlide:null,smooth:1,thresh:0.15,
      fG3:true,fG2:true,fG1:true,onlyOut:true,sortBy:'dz',q:'',field:'cosmos',
      initials:localStorage.getItem('djaqc-by')||'',decisions:dec,noteOpen:true};
    this.specCache={};
    const first=this.filtered()[0];if(first)this.state.selId=first.dja;
  }
  rng(seed){let a=seed>>>0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  gauss(r){return Math.sqrt(-2*Math.log(1-r()))*Math.cos(6.28318*r())}
  makeObjects(){
    const r=this.rng(20260718),masks=['capers-cosmos-v2','cosmos-transients-v3','cosmos-curti-v1','minerva-cosmos-p2'];
    const conf=(z,a,b)=>((1+z)*a/b)-1; // line-confusion redshift
    const objs=[];
    for(let i=0;i<100;i++){
      const zs=+(0.3+9*Math.pow(r(),1.7)).toFixed(4);
      const g=r(),grade=g<0.78?3:(g<0.92?2:1);
      let kind='fine',zp,sep=+(0.03+0.35*r()*r()).toFixed(2),ztrue=zs;
      const o=r();
      if(o<0.72){zp=zs+this.gauss(r)*0.025*(1+zs)}
      else if(o<0.845){kind='phot'; const c=r(); zp=c<0.4?conf(zs,0.65628,0.50082):(c<0.7?conf(zs,0.50082,0.65628):0.2+8*r()); } // photo-z wrong
      else if(o<0.95){kind='spec'; ztrue=+(0.3+8*Math.pow(r(),1.5)).toFixed(4); zp=ztrue+this.gauss(r)*0.03*(1+ztrue);} // spec-z (DJA) wrong
      else {kind='match'; sep=+(0.55+0.75*r()).toFixed(2); zp=0.2+8*r();}
      zp=+Math.max(0.02,zp).toFixed(4);
      objs.push({i,dja:masks[Math.floor(r()*4)]+'_'+String(10000+Math.floor(r()*89999)),
        mid:200000+Math.floor(r()*700000),ra:+(149.95+0.45*r()).toFixed(5),dec:+(2.15+0.45*r()).toFixed(5),
        sep,zs,zp,grade,ztrue,kind,mag:+(22.5+5.4*r()).toFixed(1),sn:+(kind==='match'?2+4*r():4+42*r()*r()).toFixed(1),
        slope:-0.9+1.8*r(),lstr:kind==='match'?0.05:0.25+2.6*r()*r(),f0:0.4+8*r()});
    }
    return objs;
  }
  spec(o){
    if(this.specCache[o.dja])return this.specCache[o.dja];
    const n=380,w0=0.6,w1=5.3,r=this.rng(o.i*977+11),w=new Array(n),f=new Array(n),e=new Array(n);
    const zt=o.ztrue,brk=0.3645*(1+zt),lyb=0.1216*(1+zt);
    for(let k=0;k<n;k++){
      const x=w0+(w1-w0)*k/(n-1);w[k]=x;
      let c=o.f0*Math.pow(x/2,o.slope);
      if(x<brk)c*=0.42; if(x<lyb)c*=0.02;
      let L=0;
      for(const[,lam,amp]of this.LINES){const lc=lam*(1+zt);if(lc<w0-0.1||lc>w1+0.1)continue;const s=lc/230;L+=amp*o.lstr*o.f0*Math.exp(-0.5*((x-lc)/s)**2)*2.2}
      const err=o.f0*(0.25+0.5*Math.pow(x/1.2,-1.4)+0.12*Math.pow(x/4,6))*(8/Math.max(o.sn,2));
      e[k]=err; f[k]=c+L+this.gauss(r)*err*0.55;
    }
    this.specCache[o.dja]={w,f,e}; return this.specCache[o.dja];
  }
  smoothArr(f,k){if(k<=1)return f;const n=f.length,out=new Array(n),h=Math.floor(k/2);for(let i=0;i<n;i++){let s=0,c=0;for(let j=Math.max(0,i-h);j<=Math.min(n-1,i+h);j++){s+=f[j];c++}out[i]=s/c}return out}
  nice(x){const p=Math.pow(10,Math.floor(Math.log10(x)));const m=x/p;return(m<1.5?1:m<3.5?2:m<7.5?5:10)*p}
  cur(){const{selId}=this.state;return this.objs.find(o=>o.dja===selId)||this.objs[0]}
  filtered(){
    const{thresh,fG3,fG2,fG1,onlyOut,sortBy,q}=this.state;
    let L=this.objs.filter(o=>{
      const dz=Math.abs(o.zs-o.zp)/(1+o.zs);
      if(onlyOut&&dz<=thresh)return false;
      if(!( (o.grade===3&&fG3)||(o.grade===2&&fG2)||(o.grade===1&&fG1) ))return false;
      if(q&&!(o.dja.includes(q)||String(o.mid).includes(q)))return false;
      return true});
    const key={dz:o=>-Math.abs(o.zs-o.zp)/(1+o.zs),zs:o=>o.zs,mag:o=>o.mag,sn:o=>-o.sn,sep:o=>-o.sep}[sortBy];
    return L.sort((a,b)=>key(a)-key(b));
  }
  select(dja){this.setState({selId:dja,wmin:0.6,wmax:5.3,zSlide:null})}
  step(d){const L=this.filtered();if(!L.length)return;const i=L.findIndex(o=>o.dja===this.state.selId);this.select(L[Math.min(L.length-1,Math.max(0,(i<0?0:i+d)))].dja)}
  saveDec(dec){this.setState({decisions:dec});try{localStorage.setItem('djaqc-decisions',JSON.stringify(dec))}catch(e){}}
  setDec(patch,advance){const o=this.cur();const dec={...this.state.decisions};dec[o.dja]={...(dec[o.dja]||{}),...patch,by:this.state.initials,t:Date.now()};this.saveDec(dec);
    if(advance&&(this.props.autoAdvance??true))setTimeout(()=>this.step(1),120)}
  onKey(e){
    const t=e.target.tagName;if(t==='INPUT'||t==='TEXTAREA'||t==='SELECT'||e.metaKey||e.ctrlKey)return;
    const k=e.key;
    if(k==='n'||k==='j'||k==='ArrowDown'){e.preventDefault();this.step(1)}
    else if(k==='p'||k==='k'||k==='ArrowUp'){e.preventDefault();this.step(-1)}
    else if(k==='3')this.setDec({v:'ok'},true);
    else if(k==='2')this.setDec({v:'unsure'},true);
    else if(k==='1')this.setDec({v:'bad'},true);
    else if(k==='f')this.setDec({flag:!(this.state.decisions[this.cur().dja]||{}).flag},false);
    else if(k==='0'){const dec={...this.state.decisions};delete dec[this.cur().dja];this.saveDec(dec)}
    else if(k==='r')this.setState({wmin:0.6,wmax:5.3});
    else if(k==='m'){const seq=['zspec','zphot','both','slide'];this.setState({mm:seq[(seq.indexOf(this.state.mm)+1)%4]})}
  }
  componentDidMount(){
    this._key=e=>this.onKey(e);window.addEventListener('keydown',this._key);
    this._wheel=e=>{e.preventDefault();const el=this._svg;if(!el)return;const rc=el.getBoundingClientRect();
      const fx=(e.clientX-rc.left)/rc.width*1000;const fr=Math.min(1,Math.max(0,(fx-48)/944));
      const{wmin,wmax}=this.state;const span=wmax-wmin,c=wmin+fr*span;
      const k=e.deltaY>0?1.22:1/1.22;let ns=Math.min(4.75,Math.max(0.035,span*k));
      let nmin=Math.max(0.55,Math.min(c-fr*ns,5.35-ns));this.setState({wmin:nmin,wmax:nmin+ns})};
    this.drawAll();
  }
  componentWillUnmount(){window.removeEventListener('keydown',this._key)}
  componentDidUpdate(){this.drawAll()}
  drawAll(){
    const o=this.cur(),pd=this.props.plotDark??false,{wmin,wmax}=this.state;
    const c2=this._c2;
    if(c2){const ctx=c2.getContext('2d'),{w,f}=this.spec(o);const W=1000,H=86;
      ctx.fillStyle=pd?'#26241f':'#faf9f7';ctx.fillRect(0,0,W,H);
      const img=ctx.createImageData(944,H);let fmax=0.001;
      for(let k=0;k<w.length;k++)if(w[k]>=wmin&&w[k]<=wmax)fmax=Math.max(fmax,f[k]);
      const hash=(x,y)=>{const s=Math.sin(x*12.9898+y*78.233)*43758.5453;return s-Math.floor(s)};
      for(let px=0;px<944;px++){
        const lam=wmin+(px/943)*(wmax-wmin);const t=(lam-w[0])/(w[w.length-1]-w[0])*(w.length-1);
        const i0=Math.min(w.length-2,Math.max(0,Math.floor(t)));const fv=f[i0]+(f[i0+1]-f[i0])*(t-i0);
        for(let y=0;y<H;y++){const prof=Math.exp(-((y-43)**2)/(2*6.5**2));
          let v=(fv/fmax)*prof*1.35+(hash(px,y)-0.5)*0.22;
          let g=pd?Math.min(235,25+v*260):Math.max(20,246-v*260);
          const idx=(y*944+px)*4;img.data[idx]=g;img.data[idx+1]=g;img.data[idx+2]=Math.min(255,g*(pd?0.95:1));img.data[idx+3]=255}}
      ctx.putImageData(img,48,0);
      ctx.fillStyle=pd?'#9b9797':'#7d7979';ctx.font='italic 12px Lora';ctx.fillText('2D',14,48)}
    const cc=this._cut;
    if(cc){const ctx=cc.getContext('2d'),r=this.rng(o.i*31+7),S=180;
      ctx.fillStyle='#0c0c10';ctx.fillRect(0,0,S,S);
      const id=ctx.getImageData(0,0,S,S);
      const blobs=[{x:90+(o.kind==='match'?o.sep*30:o.sep*8),y:90-(o.kind==='match'?o.sep*22:o.sep*6),a:26,q:0.55,th:r()*3.14,amp:1}];
      if(r()<0.5)blobs.push({x:20+r()*140,y:20+r()*140,a:9+r()*10,q:0.4+0.5*r(),th:r()*3.14,amp:0.3+0.4*r()});
      for(let y=0;y<S;y++)for(let x=0;x<S;x++){
        let v=(r()-0.5)*14;
        for(const b of blobs){const dx=x-b.x,dy=y-b.y,ct=Math.cos(b.th),st=Math.sin(b.th);
          const u=(dx*ct+dy*st),w2=(-dx*st+dy*ct)/b.q;const rr=Math.sqrt(u*u+w2*w2);
          v+=b.amp*225*Math.exp(-Math.pow(rr/b.a,0.62))}
        const idx=(y*S+x)*4,g=Math.min(255,Math.max(0,18+v));
        id.data[idx]=g*0.92;id.data[idx+1]=g*0.95;id.data[idx+2]=g;id.data[idx+3]=255}
      ctx.putImageData(id,0,0);
      ctx.strokeStyle='#e1ad66';ctx.lineWidth=1.4;
      ctx.beginPath();ctx.moveTo(90,68);ctx.lineTo(90,80);ctx.moveTo(90,100);ctx.lineTo(90,112);ctx.moveTo(68,90);ctx.lineTo(80,90);ctx.moveTo(100,90);ctx.lineTo(112,90);ctx.stroke();
      ctx.strokeStyle='#8ecae6';ctx.setLineDash([3,3]);ctx.beginPath();ctx.arc(blobs[0].x,blobs[0].y,14,0,6.284);ctx.stroke();ctx.setLineDash([])}
  }
  renderVals(){
    const S=this.state,o=this.cur(),pd=this.props.plotDark??false;
    const pc=pd?{bg:'#26241f',line:'#e6e1d8',err:'rgba(255,255,255,0.10)',grid:'rgba(255,255,255,0.07)',grid2:'rgba(255,255,255,0.28)',tick:'#9b9797',axis:'#605d5d'}
               :{bg:'#faf9f7',line:'#3a3733',err:'rgba(32,31,29,0.09)',grid:'rgba(32,31,29,0.06)',grid2:'rgba(32,31,29,0.30)',tick:'#7d7979',axis:'#bab6b6'};
    const dzOf=x=>Math.abs(x.zs-x.zp)/(1+x.zs);
    const list=this.filtered(),dec=S.decisions,d=dec[o.dja]||{};
    // ---- sidebar list
    const stMap={ok:['✓','var(--color-accent-700)'],unsure:['?','var(--color-neutral-600)'],bad:['✗','var(--color-neutral-800)']};
    const rows=list.map(x=>{const dd=dec[x.dja]||{};const dz=dzOf(x);
      return {name:x.dja,zs:x.zs.toFixed(3),zp:x.zp.toFixed(3),dz:dz.toFixed(3),
        dzc:dz>S.thresh?'var(--color-accent-700)':'inherit',
        gdot:x.grade===3?'var(--color-accent-500)':x.grade===2?'var(--color-neutral-400)':'var(--color-neutral-300)',
        st:(dd.flag?'⚑':'')+(stMap[dd.v]?stMap[dd.v][0]:''),stc:dd.flag?'var(--color-accent-600)':(stMap[dd.v]?stMap[dd.v][1]:'inherit'),
        bg:x.dja===o.dja?'var(--color-accent-100)':'transparent',onSel:()=>this.select(x.dja)}});
    // ---- scatter
    const zmax=Math.max(9.3,...this.objs.map(x=>Math.max(x.zs,x.zp)))+0.4;
    const sx=z=>34+z/zmax*298,sy=z=>260-z/zmax*252,t=S.thresh;
    let up='M'+sx(0)+','+sy(t/(1-t)),lo2='M'+sx(t)+','+sy(0),bd='M'+sx(0)+','+sy(t/(1-t)),bd2='M'+sx(t)+','+sy(0);
    for(let zp=0;zp<=zmax;zp+=zmax/40){const a=(zp+t)/(1-t),b=(zp-t)/(1+t);
      up+=' L'+sx(zp).toFixed(1)+','+sy(a).toFixed(1);bd+=' L'+sx(zp).toFixed(1)+','+sy(a).toFixed(1);
      if(b>=0){lo2+=' L'+sx(zp).toFixed(1)+','+sy(b).toFixed(1);bd2+=' L'+sx(zp).toFixed(1)+','+sy(b).toFixed(1)}}
    const scWedge=up+' L'+sx(0)+','+sy(zmax)+' Z '+lo2+' L'+sx(zmax)+','+sy(0)+' Z';
    const scTicks=[];for(let z=0;z<=zmax-0.5;z+=2){scTicks.push({tx:sx(z),tyy:272,anch:'middle',label:String(z),lx1:sx(z),lx2:sx(z),ly1:260,ly2:256});
      if(z>0)scTicks.push({tx:30,tyy:sy(z)+3,anch:'end',label:String(z),lx1:34,lx2:332,ly1:sy(z),ly2:sy(z)})}
    const scPts=this.objs.map(x=>{const out=dzOf(x)>t,sel=x.dja===o.dja;
      return {x:+sx(x.zp).toFixed(1),y:+sy(x.zs).toFixed(1),r:sel?5.5:(out?3.6:2.6),
        fill:x.grade===3?(out?'var(--color-accent-500)':'var(--color-neutral-400)'):'none',
        stroke:sel?'var(--color-accent-800)':(out?'var(--color-accent-600)':'var(--color-neutral-500)'),
        sw:sel?1.8:1,op:out||sel?1:0.55,onSel:()=>this.select(x.dja)}});
    // ---- 1D spectrum
    const{w,f,e}=this.spec(o);const fs=this.smoothArr(f,S.smooth);
    const{wmin,wmax}=S,span=wmax-wmin;
    const px=x=>48+(x-wmin)/span*944;
    let lo=1e9,hi=-1e9;for(let k=0;k<w.length;k++)if(w[k]>=wmin&&w[k]<=wmax){lo=Math.min(lo,fs[k]-e[k]);hi=Math.max(hi,fs[k]+e[k]*0.5)}
    if(hi<=lo){lo=0;hi=1}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;
    const py=v=>346-(v-lo)/(hi-lo)*336;
    let sp='',eu='',el='';
    for(let k=0;k<w.length;k++){if(w[k]<wmin-span*0.02||w[k]>wmax+span*0.02)continue;
      const X=px(w[k]).toFixed(1);sp+=(sp?' L':'M')+X+','+py(fs[k]).toFixed(1);
      eu+=(eu?' L':'M')+X+','+py(fs[k]+e[k]).toFixed(1);el=' L'+X+','+py(fs[k]-e[k]).toFixed(1)+el}
    const errPath=eu+el+' Z';
    const xTicks=[];{const st=this.nice(span/6);for(let x=Math.ceil(wmin/st)*st;x<=wmax;x+=st)xTicks.push({x:+px(x).toFixed(1),label:x.toFixed(st<0.1?2:1)})}
    const yTicks=[];{const st=this.nice((hi-lo)/4);for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st)yTicks.push({y:+py(v).toFixed(1),ty:+py(v).toFixed(1)+4,label:v.toFixed(st<1?1:0)})}
    const zSlideV=S.zSlide??o.zs;
    const sets=[];
    if(S.mm==='zspec'||S.mm==='both')sets.push({z:o.zs,color:'var(--color-accent-600)',dash:'',lab:0});
    if(S.mm==='zphot'||S.mm==='both')sets.push({z:o.zp,color:pd?'#9b9797':'#7d7979',dash:'5 4',lab:1});
    if(S.mm==='slide')sets.push({z:zSlideV,color:'var(--color-accent-700)',dash:'2 3',lab:0});
    const markers=[];
    for(const st of sets)for(const[nm,lam]of this.LINES){const lc=lam*(1+st.z);if(lc<wmin||lc>wmax)continue;
      markers.push({x:+px(lc).toFixed(1),color:st.color,dash:st.dash,label:nm,ty:st.lab?52:38,y1:st.lab?56:42})}
    const legend=[];let lx=60;
    for(const st of sets){const lab=st===sets[0]&&S.mm!=='zphot'?(S.mm==='slide'?'z = '+(+zSlideV).toFixed(3):'z spec '+o.zs.toFixed(3)):'z phot '+o.zp.toFixed(3);
      legend.push({x1:lx,x2:lx+22,tx:lx+27,color:st.color,dash:st.dash,label:lab});lx+=140}
    // ---- SED
    const lg=x=>Math.log10(x),sxx=x=>26+(lg(x)-lg(0.45))/(lg(5.6)-lg(0.45))*384;
    const zf=o.zp,brk=0.3645*(1+zf);
    const tmpl=x=>{let c=o.f0*Math.pow(x/2,o.slope);if(x<brk)c*=0.42;if(x<0.1216*(1+zf))c*=0.02;return c};
    let smax=0.001;const sw=[];for(let k=0;k<=120;k++){const x=0.45*Math.pow(5.6/0.45,k/120);sw.push([x,tmpl(x)]);smax=Math.max(smax,tmpl(x))}
    for(const[bn,bw]of this.BANDS)smax=Math.max(smax,tmpl(bw));
    const syy=v=>224-v/smax*190;
    const sedTemplPath=sw.map(([x,v],i)=>(i?'L':'M')+sxx(x).toFixed(1)+','+syy(v).toFixed(1)).join(' ');
    const rr=this.rng(o.i*13+5);
    const sedPts=this.BANDS.map(([bn,bw])=>{const base=this.spec(o); // photometry from true spectrum
      const t2=(bw-base.w[0])/(base.w[base.w.length-1]-base.w[0])*(base.w.length-1);const i0=Math.min(base.w.length-2,Math.max(0,Math.floor(t2)));
      let fv=base.f[i0];fv=Math.max(0.02*smax,fv*(1+this.gauss(rr)*0.06));
      const ee=smax*0.03+fv*0.07;
      return {x:+sxx(bw).toFixed(1),y:+syy(Math.min(fv,smax*1.15)).toFixed(1),e1:+syy(Math.min(fv+ee,smax*1.18)).toFixed(1),e2:+syy(Math.max(fv-ee,0)).toFixed(1),band:bn}});
    // ---- p(z) / chi2
    const zM=zmax,pzx=z=>16+z/zM*394;const sig=0.045*(1+o.zp);
    const P=z=>Math.exp(-0.5*((z-o.zp)/sig)**2)+(o.kind!=='fine'?0.14*Math.exp(-0.5*((z-o.zs)/(sig*1.4))**2):0);
    let pz='M16,228',c2p='';
    for(let k=0;k<=260;k++){const z=zM*k/260,p=P(z);pz+=' L'+pzx(z).toFixed(1)+','+(228-p*196).toFixed(1);
      const chi=Math.min(1,-2*Math.log(Math.max(p,1e-4))/18.4);c2p+=(k?' L':'M')+pzx(z).toFixed(1)+','+(32+chi*196).toFixed(1)}
    pz+=' L410,228 Z';
    const pzTicks=[];for(let z=0;z<=zM-0.4;z+=2)pzTicks.push({x:+pzx(z).toFixed(1),label:'z='+z});
    const clampT=(x)=>Math.min(392,Math.max(28,x));
    // ---- decision bar
    const decMap={ok:['✓ marked good','tag-accent'],unsure:['? marked unsure','tag-neutral'],bad:['✗ marked bad','tag-neutral']};
    const dz=dzOf(o);
    const nDone=Object.values(S.decisions).filter(x=>x.v).length;
    const set=p=>this.setState(p);
    return {
      pc,field:S.field,onField:e2=>set({field:e2.target.value}),
      thresh:S.thresh,onThresh:e2=>set({thresh:Math.max(0.02,+e2.target.value||0.15)}),
      initials:S.initials,onBy:e2=>{set({initials:e2.target.value});try{localStorage.setItem('djaqc-by',e2.target.value)}catch(x){}},
      nDone,nAll:this.objs.length,
      onExport:()=>{const H='dja_id,minerva_id,ra,dec,z_spec,z_phot,grade_dja,dz_1pz,sep_arcsec,decision,flag,z_corrected,comment,inspector\n';
        const body=this.objs.map(x=>{const dd=S.decisions[x.dja]||{};
          return [x.dja,x.mid,x.ra,x.dec,x.zs,x.zp,x.grade,dzOf(x).toFixed(4),x.sep,dd.v||'',dd.flag?1:'',dd.zNew||'',JSON.stringify(dd.comment||''),dd.by||''].join(',')}).join('\n');
        const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([H+body],{type:'text/csv'}));a.download='dja_minerva_qc_cosmos.csv';a.click()},
      scWedge,scDiag:'M'+sx(0)+','+sy(0)+' L'+sx(zmax)+','+sy(zmax),scBound:bd+' '+bd2,scTicks,scPts,
      onlyOut:S.onlyOut,tgOut:()=>set({onlyOut:!S.onlyOut}),
      fG3:S.fG3,fG2:S.fG2,fG1:S.fG1,tgG3:()=>set({fG3:!S.fG3}),tgG2:()=>set({fG2:!S.fG2}),tgG1:()=>set({fG1:!S.fG1}),
      sortBy:S.sortBy,onSort:e2=>set({sortBy:e2.target.value}),q:S.q,onQ:e2=>set({q:e2.target.value}),
      rows,noRows:rows.length===0,
      noteOpen:S.noteOpen,onNoteClose:()=>set({noteOpen:false}),
      cur:{dja:o.dja,mid:o.mid,ra:o.ra.toFixed(5),dec:o.dec.toFixed(5),sep:o.sep.toFixed(2),
        sepCol:o.sep>0.5?'var(--color-accent-700)':'inherit',mag:o.mag,sn:o.sn,grade:o.grade,
        gradeCls:o.grade===3?'tag-accent':'tag-neutral',zs:o.zs.toFixed(4),zp:o.zp.toFixed(4),dz:dz.toFixed(3),
        dzc:dz>S.thresh?'var(--color-accent-700)':'var(--color-neutral-700)',
        outLbl:dz>S.thresh?'outlier':'consistent',outBc:dz>S.thresh?'var(--color-accent-600)':'var(--color-neutral-400)'},
      mmSpec:S.mm==='zspec',mmPhot:S.mm==='zphot',mmBoth:S.mm==='both',mmSlide:S.mm==='slide',
      setMmSpec:()=>set({mm:'zspec'}),setMmPhot:()=>set({mm:'zphot'}),setMmBoth:()=>set({mm:'both'}),setMmSlide:()=>set({mm:'slide',zSlide:S.zSlide??o.zs}),
      zSlideV:+(+zSlideV).toFixed(3),onZSlide:e2=>set({zSlide:+e2.target.value}),
      sm1:S.smooth===1,sm3:S.smooth===3,sm5:S.smooth===5,setSm1:()=>set({smooth:1}),setSm3:()=>set({smooth:3}),setSm5:()=>set({smooth:5}),
      onResetZoom:()=>set({wmin:0.6,wmax:5.3}),
      svgRef:el=>{if(el&&el!==this._svg){if(this._svg)this._svg.removeEventListener('wheel',this._wheel);this._svg=el;el.addEventListener('wheel',this._wheel,{passive:false})}},
      c2Ref:el=>{this._c2=el},cutRef:el=>{this._cut=el},
      onSpecDown:e2=>{const el=this._svg;if(!el)return;const rc=el.getBoundingClientRect();const x0=e2.clientX,w0=S.wmin,spn=S.wmax-S.wmin;
        const mv=ev=>{const dw=-(ev.clientX-x0)/rc.width*1000/944*spn;let nm=Math.max(0.55,Math.min(w0+dw,5.35-spn));this.setState({wmin:nm,wmax:nm+spn})};
        const up2=()=>{window.removeEventListener('mousemove',mv);window.removeEventListener('mouseup',up2)};
        window.addEventListener('mousemove',mv);window.addEventListener('mouseup',up2)},
      specPath:sp,errPath,xTicks,yTicks,markers,legend,
      sedTemplPath,sedPts,
      pzPath:pz,chi2Path:c2p,pzZs:+pzx(o.zs).toFixed(1),pzZp:+pzx(o.zp).toFixed(1),
      pzZsT:+clampT(pzx(o.zs)).toFixed(1),pzZpT:+clampT(pzx(o.zp)).toFixed(1),pzTicks,
      onPrev:()=>this.step(-1),onNext:()=>this.step(1),
      onGood:()=>this.setDec({v:'ok'},true),onUnsure:()=>this.setDec({v:'unsure'},true),onBad:()=>this.setDec({v:'bad'},true),
      onFlag:()=>this.setDec({flag:!d.flag},false),
      decLbl:(d.flag?'⚑ ':'')+(decMap[d.v]?decMap[d.v][0]:'unreviewed'),decCls:d.v?decMap[d.v][1]:'tag-outline',
      curZNew:d.zNew??'',onZNew:e2=>this.setDec({zNew:e2.target.value===''?null:+e2.target.value},false),
      curComment:d.comment||'',onComment:e2=>this.setDec({comment:e2.target.value},false),
      kbdDisp:(this.props.showHints??true)?'inline-flex':'none',
    };
  }
}
