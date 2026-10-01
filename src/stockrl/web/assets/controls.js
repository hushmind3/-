const baseRenderControls=renderControls;
renderControls=function(d){
 baseRenderControls(d);
 const changing=!!(d.stopping||d.restarting),observe=!!d.observe_enabled,paper=!!d.paper_enabled,learning=d.learning_enabled!==false;
 $("serverRestartBtn").disabled=busy||changing;
 for(const [id,on,label] of [["observeBtn",observe,"모델 판단"],["paperBtn",paper,"가상계좌 체결"],["learningBtn",learning,"replay 학습"]]){
  text(id,label+": "+(on?"ON":"OFF"));$(id).classList.toggle("active",on);$(id).setAttribute("aria-pressed",String(on));$(id).disabled=busy||changing;
 }
 text("runTitle",d.running?observe?"모델 판단 실행 중":"모델 판단 중지 · 시세 저장 계속":"시스템 정지");
 text("runDetail","판단 "+(observe?"ON":"OFF")+" | 체결 "+(paper?"ON":"OFF")+" | 학습 "+(learning?"ON":"OFF")+" | 실제 주문 OFF");
 text("autonomyDetail","판단: 두 모델의 새 추론·승급전. 체결: 기존 주문 실행·새 판단 주문 접수. 학습: 저장된 replay로 두 모델 업데이트. 각각 독립 제어합니다. 판단 OFF에서도 시세 저장·기존 보유 평가·replay 학습은 계속됩니다. 비트코인·환율 등 문맥용 시세만 바뀌면 풀 추론하지 않습니다.");
};

$("observeBtn").onclick=()=>runCommand($("observeBtn"),async()=>{const enabled=!current?.observe_enabled;await api("/api/modes",{observe_enabled:enabled});feedback(enabled?"두 모델의 판단을 켰습니다.":"두 모델의 새 판단·승급전을 중지합니다. 시세 저장과 replay 학습은 계속됩니다.")});
$("paperBtn").onclick=()=>runCommand($("paperBtn"),async()=>{const enabled=!current?.paper_enabled;await api("/api/modes",{paper_enabled:enabled});feedback(enabled?"가상계좌 체결을 켰습니다. 판단·학습 설정은 유지됩니다.":"새 가상 체결을 중지합니다. 보유 평가와 학습 설정은 유지됩니다.")});
$("learningBtn").onclick=()=>runCommand($("learningBtn"),async()=>{const enabled=current?.learning_enabled===false;await api("/api/modes",{learning_enabled:enabled});feedback(enabled?"두 모델의 replay 학습을 켰습니다.":"현재 학습 batch 저장 후 다음 batch부터 멈춥니다. 미학습 경험은 보존됩니다.")});
