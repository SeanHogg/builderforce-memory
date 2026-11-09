import { foldRepeatedNarration, NARRATION_OVERLAP, PROGRESS_NOTE } from './fold.js';
const readCall=(id:string,p:string)=>({id,function:{name:'read_file',arguments:JSON.stringify({path:p})}});
const editCall=(id:string,p:string)=>({id,function:{name:'edit_file',arguments:JSON.stringify({path:p})}});
it('debug',()=>{
 const msgs:any[]=[
  {role:'assistant',content:'Reading the config to find the lever wiring.',tool_calls:[readCall('r1','cfg.ts')]},
  {role:'tool',tool_call_id:'r1',content:'lever config text'},
  {role:'assistant',content:'Reading the config to find the lever wiring once more.',tool_calls:[editCall('e1','cfg.ts')]},
  {role:'tool',tool_call_id:'e1',content:{ok:true}},
 ];
 const b7=foldRepeatedNarration(msgs);
 console.log('b7[2]=',JSON.stringify(b7[2].content),'NOTE=',JSON.stringify(PROGRESS_NOTE));
 const words=(t:string)=>new Set(t.toLowerCase().split(/[^a-z0-9_]+/).filter((w:string)=>w.length>0));
 const a='Reading the config to find the lever wiring.';
 const b='Reading the config to find the lever wiring once more.';
 console.log('setA',[...words(a)],'setB',[...words(b)]);
 console.log('threshold',NARRATION_OVERLAP);
});
