setClock();
setInterval(setClock,1000);
refresh();
setInterval(()=>{if(!document.hidden)refresh()},5000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden)refresh()});
