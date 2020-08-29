$(document).ready(function() {
    //Set current year
    $('#year').text(new Date().getFullYear());

    //Change gimmick image on hover
    $('#gimmick')
    .mouseover(function () {$(this).attr("src", "projects/star_runner/player.gif");})
    .mouseout(function () {$(this).attr("src", "projects/star_runner/player.png");});
});