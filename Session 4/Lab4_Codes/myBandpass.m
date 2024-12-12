function X = myBandpass(sig, lower_edge, higher_edge, fs)
N = 4;
Apass = 1;
h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, lower_edge, higher_edge, Apass, fs);
Hd = design(h, 'cheby1');
for c = 1 : 30
    X(c,:) = filter(Hd, sig(c,:));
end
end