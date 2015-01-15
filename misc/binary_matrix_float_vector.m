m = 10000;
n = 5000;

time = zeros(3, 11);

for rho = 0.0:0.1:1.0
    A = logical(binornd(1, rho, m, n));
    x = rand(n, 1);
    i = floor(rho * 10) + 1;
    % multiply directly
    t0 = tic;
    b = A*x;
    time(1, i) = toc(t0);
    % sparse
    t0 = tic;
    b = sparse(A) * sparse(x);
    time(2, i) = toc(t0);
    % bsxfun
    t0 = tic;
    b = sum(bsxfun(@times, A, x'), 2);
    time(3, i) = toc(t0);
end

rhos = 0:0.1:1.0;
plot(rhos, time(1, :), 'ro-'); hold on;
plot(rhos, time(2, :), 'gx-');
plot(rhos, time(3, :), 'bv-');
legend('mul', 'sparse', 'bsxfun');
hold off;