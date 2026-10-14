"""Self-contained decorative loader; status copy uses native Streamlit text."""

LOADING_GRAPHIC = """
<style>
.law-loading-art{display:flex;align-items:center;gap:16px;width:100%;max-width:520px;padding:6px 0 8px;box-sizing:border-box;}
.law-loading-art .law-loading-bubbles{position:relative;display:block;flex:0 0 30px;width:30px;height:34px;overflow:hidden;border-radius:10px;background:#f5ebd8;}
.law-loading-art .law-loading-bubbles i{position:absolute;bottom:4px;width:5px;height:5px;border-radius:50%;background:#e2be7a;animation:law-loader-rise 1.8s ease-in-out infinite;}
.law-loading-art .law-loading-bubbles i:nth-child(1){left:5px;}.law-loading-art .law-loading-bubbles i:nth-child(2){left:13px;animation-delay:.5s;}.law-loading-art .law-loading-bubbles i:nth-child(3){left:21px;animation-delay:1s;}
.law-loading-art .law-loading-bars{display:flex;flex:1;min-width:0;flex-direction:column;gap:10px;}
.law-loading-art .law-loading-bars>span{display:block;flex-shrink:0;height:8px;border-radius:5px;background:linear-gradient(100deg,#f5ebd8 15%,#f5ebd8 45%,#f5ebd8 70%);background-size:250% 100%;animation:law-loader-shimmer 2.4s ease-in-out infinite;}
@keyframes law-loader-rise{0%,100%{transform:translateY(0);opacity:.3;}65%{transform:translateY(-22px);opacity:1;}}
@keyframes law-loader-shimmer{0%{background-position:100% 0;}100%{background-position:-100% 0;}}
@media(prefers-reduced-motion:reduce){.law-loading-art .law-loading-bubbles i,.law-loading-art .law-loading-bars>span{animation:none;}}
</style>
<div class="law-loading-art" aria-hidden="true" style="display:flex;align-items:center;gap:16px;max-width:520px;padding:6px 0 8px">
  <span class="law-loading-bubbles" style="position:relative;display:block;flex:0 0 30px;height:34px;border-radius:10px;background:#f5ebd8;overflow:hidden"><i></i><i></i><i></i></span>
  <div class="law-loading-bars" style="display:flex;flex:1;min-width:0;flex-direction:column;gap:10px">
    <span style="display:block;width:95%;height:8px;border-radius:5px;background-color:#f5ebd8"></span>
    <span style="display:block;width:65%;height:8px;border-radius:5px;background-color:#f5ebd8"></span>
  </div>
</div>
"""
