"""로딩 장식 그래픽. 진행 안내 문구는 기본 Streamlit 텍스트 위젯에서 표시한다."""

# st.html의 출력 범위 안에 스타일을 함께 전달한다.
# 외부 CSS의 적용 시점과 무관하게 그려지도록 주요 레이아웃의 인라인 스타일도 유지한다.
LOADING_GRAPHIC = """
<style>
.law-loading-art{display:flex;align-items:center;gap:16px;width:100%;max-width:520px;padding:6px 0 8px;box-sizing:border-box;}
.law-loading-art .law-loading-bubbles{position:relative;display:block;flex:0 0 30px;width:30px;height:34px;overflow:hidden;border-radius:10px;background:#1e293b;}
.law-loading-art .law-loading-bubbles i{position:absolute;bottom:4px;width:5px;height:5px;border-radius:50%;background:#e2be7a;animation:law-loader-rise 1.8s ease-in-out infinite;}
.law-loading-art .law-loading-bubbles i:nth-child(1){left:5px;}.law-loading-art .law-loading-bubbles i:nth-child(2){left:13px;animation-delay:.5s;}.law-loading-art .law-loading-bubbles i:nth-child(3){left:21px;animation-delay:1s;}
.law-loading-art .law-loading-bars{display:flex;flex:1;min-width:0;flex-direction:column;gap:10px;}
.law-loading-art .law-loading-bars>span{display:block;flex-shrink:0;height:8px;border-radius:5px;background:linear-gradient(100deg,#f8fafc14 15%,#e2be7a40 45%,#f8fafc14 70%);background-size:250% 100%;animation:law-loader-shimmer 2.4s ease-in-out infinite;}
@keyframes law-loader-rise{0%,100%{transform:translateY(0);opacity:.3;}65%{transform:translateY(-22px);opacity:1;}}
@keyframes law-loader-shimmer{0%{background-position:100% 0;}100%{background-position:-100% 0;}}
@media(prefers-reduced-motion:reduce){.law-loading-art .law-loading-bubbles i,.law-loading-art .law-loading-bars>span{animation:none;}}
</style>
<div class="law-loading-art" aria-hidden="true" style="display:flex;align-items:center;gap:16px;max-width:520px;padding:6px 0 8px">
  <span class="law-loading-bubbles" style="position:relative;display:block;flex:0 0 30px;height:34px;border-radius:10px;background:#1e293b;overflow:hidden"><i></i><i></i><i></i></span>
  <div class="law-loading-bars" style="display:flex;flex:1;min-width:0;flex-direction:column;gap:10px">
    <span style="display:block;width:95%;height:8px;border-radius:5px;background-color:#f8fafc24"></span>
    <span style="display:block;width:65%;height:8px;border-radius:5px;background-color:#f8fafc24"></span>
  </div>
</div>
"""
