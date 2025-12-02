$out_dir = 'build';
$pdf_mode = 5;
$pdflatex = 'pdflatex -interaction=nonstopmode -output-directory=' . $out_dir . ' %O %S';
system("mkdir -p $out_dir") unless -d $out_dir;

END {
    my ($base, $path, $ext) = fileparse($ARGV[-1], qr/\.[^.]*/);
    my $pdf_source = "$out_dir/$base.pdf";
    my $pdf_dest = "$base.pdf";
    system("cp '$pdf_source' '$pdf_dest'") if -e $pdf_source && $pdf_source ne $pdf_dest;
}
