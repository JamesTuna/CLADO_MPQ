# echo "1kx512 experiment"
# python3 clado_search_i1k_MIQCP1.py &> clado_search_i1k_MIQCP1.log
# echo "1kx256 experiment"
# python3  clado_search_i1k_MIQCP2.py &> clado_search_i1k_MIQCP2.log
echo "1kx512(2) experiment"
python3 clado_gc_i1k.py &> clado_search_i1k_MIQCP1_2.log
echo "1kx512(2) search"
python3 clado_search_i1k_MIQCP1.py &> clado_search_i1k_MIQCP1_2.log

