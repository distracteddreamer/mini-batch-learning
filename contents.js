var lists = document.getElementsByTagName('ul')
ulArray = Array.from(lists)
sortArray = function(ul){
  x = ul.getElementsByTagName('li');
  y = Array.from(x);
  y.sort(
    function(a, b){
      aText = a.innerText;
      bText = b.innerText;
      if(aText > bText) return 1;
      if(aText < bText) return -1;
      return 0;
    });
  ul.innerHTML = '';
  for(i=0;i<y.length; i++){
    ul.appendChild(y[i]);
  }
}
ulArray.forEach(sortArray);