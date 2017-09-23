<?php
header('Content-Type: application/json; charset=UTF-8');
setlocale(LC_CTYPE, "en_US.UTF-8");
$data = json_decode(file_get_contents('php://input'));
$text = $data->{'text'};
$text = addslashes($text);

$cloudsight = $data->{'cloudsight'};
$cloudsight = addslashes($cloudsight);


$o = shell_exec('python3.4 /var/www/html/BM_nor_eng/BM_main.py ' .  escapeshellarg($text) . ' ' . escapeshellarg($cloudsight));
echo $o;




?>