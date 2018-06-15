function [ vec ] = noise_combination( num, min_l, max_l )
%NOISE_COMBINATION Summary of this function goes here
%   Detailed explanation goes here
%   num: number of noise level for each distortion
%   min_l: min sum of levels
%   max_l: max sum of levels
%   vec: a matrix, each row of which is a possible combination

vec = [];
for k = 1:num
    for m = k:num
        for n = m:num
            if k+m+n-2 >= min_l && k+m+n-2 <= max_l
                if k == n
                    vec(end+1, 1:3) = [k,m,n];
                elseif k == m
                    vec(end+1: end+3, 1:3) = [k,m,n; k,n,m; n,k,m];
                elseif m == n
                    vec(end+1: end+3, 1:3) = [k,m,n; m,k,n; m,n,k];
                else
                    vec(end+1:end+6, 1:3) = [k,m,n; k,n,m; m,k,n; m,n,k;...
                        n,k,m; n,m,k];
                end
            end
        end
    end
end



end

