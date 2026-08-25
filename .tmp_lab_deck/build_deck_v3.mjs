import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
const OUT="/Users/virginiaceccatelli/Documents/semantic-flow/semantic_flow_experiments_explained.pptx";
const PRE="/Users/virginiaceccatelli/Documents/semantic-flow/.tmp_lab_deck/rendered_v3";
const ROOT="/Users/virginiaceccatelli/Documents/semantic-flow";
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:"#101114",gray:"#626875",pale:"#F1F2F4",rule:"#C8CBD0",blue:"#3D8DFF",cyan:"#6DCBF4",light:"#D0EDFA",white:"#FFFFFF",green:"#258A60",red:"#A23B3B"};
function box(s,x,y,w,h,fill=C.pale){return s.shapes.add({geometry:"rect",position:{left:x,top:y,width:w,height:h},fill,line:{style:"solid",fill:"none",width:0}})}
function txt(s,t,x,y,w,h,z=24,o={}){const q=s.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});q.text=t;q.text.style={fontSize:z,typeface:"Helvetica Neue",color:o.color||C.ink,bold:!!o.bold,alignment:o.align||"left",verticalAlignment:o.valign||"top",autoFit:"shrinkText",insets:{top:0,right:0,bottom:0,left:0}};return q}
function title(s,t,n){txt(s,t,42,30,1160,64,39,{bold:true});txt(s,String(n).padStart(2,"0"),1185,665,48,20,14,{color:C.gray,align:"right"})}
function notes(s,a,x=""){s.speakerNotes.textFrame.setText(`${x}${x?"\n\n":""}[Sources]\n${a.map(v=>`- ${v}`).join("\n")}`)}
async function img(s,path,pos,alt){const b=await fs.readFile(path);s.images.add({blob:b,contentType:"image/png",alt,fit:"contain",position:pos})}
function bar(s,pos,cats,series,max=1){s.charts.add("bar",{position:pos,categories:cats,series,hasLegend:true,legend:{position:"bottom",overlay:false},dataLabels:{showValue:true,numberFormat:"0%",position:"outEnd"},chartFill:C.white,chartLine:{style:"solid",width:0,fill:C.white},plotAreaFill:{type:"none"},plotAreaLine:{style:"solid",width:0,fill:C.white},xAxis:{visible:true,deleted:false,line:{style:"solid",width:1,fill:C.rule}},yAxis:{visible:true,deleted:false,min:0,max,numberFormatCode:"0%",majorGridlines:{style:"solid",fill:C.rule,width:1}}})}

// 1: instrument map
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"Probe, R-lens and DAS answer different questions",1);
 const xs=[42,452,862], heads=["PROBE","R-LENS","DAS"], qs=["Is the information present?","Where is an output score attributed?","Does changing the representation change the output?"], acts=["Train a separate linear classifier on hidden states.","Propagate one model output score backward to earlier tokens.","Interchange one learned semantic coordinate between runs."], out=["Availability","Output-associated attribution","Causal use"];
 for(let i=0;i<3;i++){box(s,xs[i],140,376,435,i===2?C.light:C.pale);txt(s,heads[i],xs[i]+25,168,320,28,19,{bold:true,color:i===2?C.blue:C.gray});txt(s,qs[i],xs[i]+25,222,320,75,27,{bold:true});txt(s,acts[i],xs[i]+25,330,320,105,21);txt(s,out[i],xs[i]+25,500,320,30,20,{bold:true,color:i===2?C.blue:C.gray})}
 txt(s,"A positive result from one method does not imply a positive result from the others.",42,615,1120,40,25,{bold:true});
 notes(s,["docs/METHODS.md §§0, 3, 6, 8."]);
}

// 2 probe
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"Taint flow is linearly decodable at the sink",2);
 txt(s,"Experiment",42,112,135,26,19,{bold:true,color:C.gray});txt(s,"Train on hidden states at the sink argument: source-derived or trusted? Freeze the probe; test held-out programs.",180,106,1010,52,23,{bold:true});
 await img(s,`${ROOT}/results/figures/sinkflow_cells_deepseek-coder-6.7b.png`,{left:42,top:175,width:780,height:430},"Probe accuracy across sink families, structures and obfuscations for DeepSeek-Coder 6.7B");
 txt(s,"Clean",865,205,120,25,19,{bold:true,color:C.gray});txt(s,"1.00",1030,186,160,60,48,{bold:true,color:C.blue,align:"right"});
 txt(s,"Lexical floors",865,292,170,25,19,{bold:true,color:C.gray});txt(s,"0.43–0.54",1010,279,180,48,32,{bold:true,align:"right"});
 txt(s,"Meaning",865,380,120,25,19,{bold:true,color:C.blue});txt(s,"The sink state contains information that separates source-derived from trusted flow.",865,420,325,105,22,{bold:true});
 txt(s,"Not shown",865,557,120,25,18,{bold:true,color:C.gray});txt(s,"Whether the model uses that information to produce an answer.",865,590,325,55,19,{color:C.gray});
 notes(s,["results/figures/sinkflow_cells_deepseek-coder-6.7b.png.","results/tables/sinkflow_clean_deepseek-coder-6.7b.csv.","docs/RESULTS.md R5."],"Figure is a repository-generated experimental result. Clean held-out hidden-state readout reaches 1.00 at the reported cell; local and whole-program lexical floors are approximately 0.43–0.54 across models.");
}

// 3 robustness
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"Frozen probes fail most under control-flow flattening",3);
 txt(s,"Experiment",42,108,130,24,18,{bold:true,color:C.gray});txt(s,"Apply clean-trained binding and def–use probes after semantics-preserving rewrites.",175,103,960,38,22,{bold:true});
 await img(s,`${ROOT}/results/figures/obfuscation_binding_deepseek-coder-6.7b.png`,{left:42,top:155,width:790,height:480},"Binding probe accuracy by layer and obfuscation for DeepSeek-Coder 6.7B");
 txt(s,"DeepSeek 6.7B",875,170,290,30,22,{bold:true});txt(s,"Best-layer binding",875,225,220,24,18,{color:C.gray});
 const rr=[["Rename","0.90"],["Opaque","0.86"],["MBA encode","0.86"],["Flatten","0.77"]];rr.forEach((r,i)=>{const y=270+i*52;if(i===3)box(s,855,y-8,335,42,C.light);txt(s,r[0],875,y,165,26,20,{bold:i===3});txt(s,r[1],1080,y,90,26,21,{bold:true,align:"right"})});
 txt(s,"Across 3 models",875,505,200,24,18,{bold:true,color:C.blue});txt(s,"Flattening is the lowest-accuracy transformation for both binding and def–use.",875,540,300,84,21,{bold:true});
 notes(s,["results/figures/obfuscation_binding_deepseek-coder-6.7b.png.","results/tables/obfuscation_robustness_*.csv.","docs/RESULTS.md R4."],"Heatmap columns are cumulative obfuscation levels 0–4; rows are model layers. The side numbers summarize atomic transformations at the best layer.");
}

// 4 R lens mechanism
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"R-lens attributes one answer score across the input",4);
 box(s,42,145,330,390,C.pale);txt(s,"Model output",67,175,280,28,19,{bold:true,color:C.gray});txt(s,"score(‘vulnerable’)",67,225,280,45,28,{bold:true});txt(s,"R-lens starts here",67,304,280,28,20,{bold:true,color:C.blue});txt(s,"It changes only the backward calculation; the forward answer is unchanged.",67,352,275,100,21);
 box(s,475,145,330,390,C.light);txt(s,"Backward allocation",500,175,280,28,19,{bold:true,color:C.blue});txt(s,"Conserving relevance rules propagate the score through each layer.",500,225,275,110,23,{bold:true});txt(s,"Earlier-token shares add back to approximately the original score.",500,375,275,90,21);
 box(s,908,145,330,390,C.pale);txt(s,"Grouped result",933,175,280,28,19,{bold:true,color:C.gray});txt(s,"taint chain\ntrust chain\nsink\nother tokens",933,225,280,160,26,{bold:true});txt(s,"Compare role shares across a matched safe/unsafe pair.",933,420,275,80,21);
 txt(s,"R-lens does not test whether information is decodable, and it does not establish causal necessity.",42,600,1160,42,24,{bold:true});
 notes(s,["docs/METHODS.md §§6.4–6.5.","docs/RESULTS.md R6 and R9."],"The R-lens gate passes on both DeepSeek models: final-layer agreement 1.0000; median conservation error 0.0000 / 0.0001 across validation layers. It is not applicable to StarCoder2 under the implemented rules.");
}

// 5 R lens result
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"R-lens finds a small redistribution between identical chains",5);
 txt(s,"Matched-pair test",42,108,170,24,18,{bold:true,color:C.gray});txt(s,"The taint and trust chains use identical tokens; only which chain reaches the sink changes.",215,103,980,38,22,{bold:true});
 txt(s,"Reported cell",42,170,280,24,17,{bold:true,color:C.gray});txt(s,"DeepSeek 1.3B · L0",420,170,300,24,19,{bold:true});txt(s,"DeepSeek 6.7B · L11",820,170,330,24,19,{bold:true});
 const lab=["Taint chain shifts down","Trust chain shifts up","Median taint shift","Sign-test p","Mean permutation p"],a=["65 / 72","63 / 72","−0.014","7.0 × 10⁻¹³","0.57  (fails)"],b=["64 / 72","57 / 72","−0.021","5.8 × 10⁻¹²","< 0.001"];
 lab.forEach((v,i)=>{const y=225+i*61;if(i%2===0)box(s,30,y-10,1170,46,C.pale);txt(s,v,42,y,300,26,19,{bold:true});txt(s,a[i],420,y,280,26,22,{bold:i<2,color:i===4?C.gray:C.ink});txt(s,b[i],820,y,320,26,22,{bold:i<2})});
 box(s,42,542,1158,1,C.rule);txt(s,"What it shows",42,570,155,24,18,{bold:true,color:C.blue});txt(s,"Changing the data-flow relation changes output attribution beyond token identity.",205,566,970,34,22,{bold:true});
 txt(s,"Strength",42,615,90,22,17,{bold:true,color:C.gray});txt(s,"Weak: 1–2% of the answer score; observational; depth does not replicate across scale.",140,611,1000,28,18,{color:C.gray});
 txt(s,"Open",42,653,70,22,17,{bold:true,color:C.gray});txt(s,"The attribution does not trace a simple ‘follow the active taint chain’ mechanism.",115,649,1000,28,18,{bold:true});
 notes(s,["results/tables/relevance_summary_deepseek-coder-1.3b.csv.","results/tables/relevance_summary_deepseek-coder-6.7b.csv.","docs/RESULTS.md R9."],"Attribution conservation at the reported cells has median |rho−1| around 1e−7 / 2e−7. The direction is counterintuitive rather than a simple active-chain trace, so the claim is redistribution, not faithful reasoning-path recovery.");
}

// 6 yes/no
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"Yes/no taint question: poor decisions, informative margins",6);
 txt(s,"Prompt",42,118,100,24,18,{bold:true,color:C.gray});txt(s,"“Is the value passed to the sink tainted? Answer yes or no.”",145,112,930,38,23,{bold:true});
 bar(s,{left:42,top:190,width:730,height:405},["DeepSeek 1.3B","DeepSeek 6.7B","StarCoder2 3B"],[{name:"Final accuracy",values:[.5,.5,.5],fill:C.gray},{name:"Correct pair ordering",values:[.694,.75,.917],fill:C.blue}]);
 box(s,820,185,375,155,C.pale);txt(s,"Final choice",845,210,140,24,18,{bold:true,color:C.gray});txt(s,"50%",1040,192,125,58,45,{bold:true,align:"right"});txt(s,"Models collapse to one answer token: always ‘no’ or always ‘yes’.",845,267,320,60,20);
 box(s,820,370,375,180,C.light);txt(s,"Margin ordering",845,395,190,24,18,{bold:true,color:C.blue});txt(s,"69–92%",1000,379,165,55,38,{bold:true,color:C.blue,align:"right"});txt(s,"Unsafe programs usually move the yes−no score in the correct direction relative to matched safe programs.",845,455,320,75,20,{bold:true});
 txt(s,"Interpretation",42,630,135,24,18,{bold:true,color:C.blue});txt(s,"The distinction affects output scores, but the decision threshold is badly calibrated. This is not reliable task performance.",180,625,1015,38,20,{bold:true});
 notes(s,["results/tables/positive_behaviour_summary_*.csv.","docs/RESULTS.md R8."],"Clean held-out sink prompt: pair ordering 0.694 / 0.750 / 0.917. At the strongest internal cell, lens sign consistency is approximately 0.889 / 0.847 / 0.944. The logit, J- and R-lenses reach essentially the same conclusion; R-lens is not required for this result.");
}

// 7 DAS design
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"DAS asks whether a semantic coordinate controls the output",7);
 txt(s,"Binding experiment",42,110,180,24,18,{bold:true,color:C.gray});txt(s,"Learn one direction that distinguishes which definition a use refers to; interchange only that coordinate.",225,104,970,46,23,{bold:true});
 txt(s,"Factorial control",42,190,190,24,20,{bold:true,color:C.blue});txt(s,"The same binding change requires opposite answer-token movements in the two arms.",235,185,930,38,22,{bold:true});
 txt(s,"Account",42,267,300,24,17,{bold:true,color:C.gray});txt(s,"Train arm ab",475,267,210,24,17,{bold:true,color:C.gray});txt(s,"Held-out arm ba",800,267,240,24,17,{bold:true,color:C.gray});
 const rows=[["Binding representation","positive","positive"],["Fixed token / answer direction","positive","negative"],["Required DAS result","positive","positive"]];rows.forEach((r,i)=>{const y=310+i*72;if(i!==1)box(s,30,y-10,1110,54,i===2?C.light:C.pale);txt(s,r[0],42,y,360,28,21,{bold:i===2});txt(s,r[1],475,y,180,28,22,{bold:true,color:C.green});txt(s,r[2],800,y,180,28,22,{bold:true,color:i===1?C.red:C.green})});
 txt(s,"Intervention",42,555,130,24,18,{bold:true,color:C.gray});txt(s,"rank 1 · variable-use site · no dose parameter",175,550,540,34,22,{bold:true});
 txt(s,"Protocol",42,612,100,24,18,{bold:true,color:C.gray});txt(s,"120 calibration bases · 280 held-out test bases · full-vocabulary argmax",145,607,910,34,21);
 txt(s,"Models",42,659,90,22,17,{bold:true,color:C.gray});txt(s,"DeepSeek-Coder 6.7B (L8) · StarCoder2 3B (L11)",135,655,760,28,19);
 notes(s,["docs/METHODS.md §8.","docs/RESULTS.md R10."],"The learned subspace is fitted only on arm ab. Successful transfer to ba falsifies a fixed answer-token direction because that account predicts the opposite sign in ba.");
}

// 8 DAS result
{
 const s=deck.slides.add();s.background.fill=C.white;title(s,"Rank-1 binding interchange controls the answer",8);
 txt(s,"Installed-binding emission · held-out arm ba · 560 rows / 280 bases per model",42,105,1080,24,17,{color:C.gray});
 bar(s,{left:42,top:150,width:760,height:470},["DAS rank-1","Mean difference","Answer direction","Random, dose-matched"],[{name:"DeepSeek 6.7B",values:[1,.768,.043,.018],fill:C.blue},{name:"StarCoder2 3B",values:[1,.545,.184,.304],fill:C.cyan}]);
 box(s,850,160,350,125,C.light);txt(s,"100%",875,178,300,52,46,{bold:true,color:C.blue});txt(s,"both models · both arms",875,238,300,26,20,{bold:true});
 txt(s,"Edit magnitude",850,330,160,24,18,{bold:true,color:C.gray});txt(s,"DAS: 47.9% of hidden-state norm",850,365,340,50,22,{bold:true});txt(s,"Mean difference: 71.0% / 71.9%",850,420,340,30,19);
 txt(s,"Controls",850,485,100,24,18,{bold:true,color:C.gray});txt(s,"6/6 gates · 14/14 machinery checks · 0% non-candidate outputs",850,520,340,64,20,{bold:true});
 txt(s,"Licensed conclusion",850,616,165,24,18,{bold:true,color:C.blue});txt(s,"Binding information is not only decodable: changing it changes the model's output.",1015,610,185,54,18,{bold:true});
 notes(s,["results/binding/deepseek-coder-6.7b/interchange_summary.csv.","results/binding/starcoder2-3b/interchange_summary.csv.","docs/RESULTS.md R10."],"This is causal evidence for binding, not yet for taint flow. A taint-specific intervention would be needed to establish causal use of the source-to-sink distinction.");
}

await fs.mkdir(PRE,{recursive:true});for(const[i,s]of deck.slides.items.entries()){const p=await deck.export({slide:s,format:"png",scale:1});await fs.writeFile(`${PRE}/slide-${i+1}.png`,new Uint8Array(await p.arrayBuffer()));const l=await s.export({format:"layout"});await fs.writeFile(`${PRE}/slide-${i+1}.layout.json`,await l.text())}const m=await deck.export({format:"webp",montage:true,scale:1});await fs.writeFile(`${PRE}/montage.webp`,new Uint8Array(await m.arrayBuffer()));const f=await PresentationFile.exportPptx(deck);await f.save(OUT);
