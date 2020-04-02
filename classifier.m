%classification of lamp and mandolin with one-layer nn

function classifier()
    clear all; 
    close all; 
    clc;
    
    %teaching-testing ratio
    %az adathalmaz mekkora reszet hasznaljuk tanitasra illetve tesztelesre
    ttRatio = 0.5;
    
    %learning rate
    lr = 0.0055;

    %log-sigmoid aktivacios fuggveny
    %f = @logsig;
    %gr = @(x) f(x).*(1-f(x));
    
    %tangens hiperbolikusz aktivacios fuggveny
    f = @tanh;
    gr = @(x) 1-f(x).^2;
    
    %beolvassa az adathalmazokat es felosztja oket az aranynak megfeleloen
    [xTrain, yTrain, xTest, yTest, imgTest] = loadData('lamp', 'mandolin', ttRatio);
    
    % tanito fv negyzetes hibafuggvennyel
    [w, E] = offlineLearning(xTrain, yTrain, f, gr, lr, @stoppingCondition);

    
    %abrazoljuk a rendszer hibajanak a valtozasat
    figure
    hold on;
    ind = [1:1:length(E)];
    plot(E(:),ind);
    set(gcf, 'Position', [710,300,610,536]);


    y = testing(xTest, f, w);
    
    y = y > (max(yTest(:))/2);
    
    displayResult(y, yTest, imgTest);
end




%%
function [xTrain, yTrain, xTest, yTest, imageTest] = loadData(folder1, folder2, ttRatio)
    [images1, numberOfImages1] = preprocData(folder1);
    [images2, numberOfImages2] = preprocData(folder2);

    % ket adathalmaz osszefuzese
    images = [images1; images2];
    numberOfImages = numberOfImages1 + numberOfImages2;
    
    % bemeneti tomb random permutalasa
    d = [zeros(numberOfImages1, 1); ones(numberOfImages2,1)];
    p = randperm(numberOfImages);
    d = d(p);

    X = zscore(images(p,:));
    
    % tanitasi halmaz merete n
    n = round(numberOfImages * ttRatio);
    
    % adathalmaz felosztasa tanitasi es tesztelesi reszre
    xTrain = X(1:n,:);
    xTest = X(n+1:end,:);
    yTrain = d(1:n);
    yTest = d(n+1:end);
    
    imageTest = images(p(n+1:end),:);
end




%%
%betoltes es elofeldolgozas (szin + meret)
function [images, numberOfImages] = preprocData(folder)
    files = dir(['/home/tunde/Linux/Documents/Harmadev-II/ai/delta/' folder '/*.jpg']);
    numberOfImages = length(files);
    images = [];
    
    % kepek beolvasasa, feldolgozasa
    for i = 1:numberOfImages
        fprintf('loading %s\n',files(i).name);
        image_i = imread(['/home/tunde/Linux/Documents/Harmadev-II/ai/delta/' folder '/' files(i).name]);
        
        %atmeretezes, szincsatorna redukalasa
        image_i = imresize(image_i, [64, 64]);
        if size(image_i,3) == 3
            image_i = rgb2gray(image_i);
        end
        
        images(i,:) = double(image_i(:))';
    end
end



%%
% teszthalmaz tesztelese
function y = testing(x, f, w)
    % size(x, 1) - sorok szama x
    % size(x, 2) - oszlopok szama x
    
    %inicializalas
    y = zeros(size(x,1), size(w,2));

    for i = 1:size(x,1)
        y(i,:) = f((x(i,:)*w));
    end
end



%%
function displayResult(predicted, actual, X)
    labels = ["lamp", "mandolin"];
    % kiabrazolando eredmenyek dimenzioja
    nrows = 6;
    ncols = 6;
    % feltetelezett ertekek (eltolva 1-el jobbra)
    predicted = predicted + 1;
    % aktualis ertekek (eltolva 1-el jobbra)
    actual = actual + 1;
    figure

    % eredmeny abrazolasa
    for i = 1:nrows
        for j = 1:ncols
            k = (i-1)*ncols + j;
            subplot(nrows, ncols, k);
            image_i = uint8(reshape(X(k,:), 64, 64));
            imshow(image_i);
            % feltetelezett csoport cimkeje
            xlabel(labels(predicted(k)));
        end
    end
    
    % eredmenyek abrazolasa es pozocionalasa
    set(gcf, 'Position', [50, 211, 560, 790]);
    set(gcf(), 'MenuBar', 'none');
    
    figure
    % konfizios matrix abrazolasa es pozocionalasa
    confusionchart(labels(actual), labels(predicted));
    set(gcf, 'Position', [710,0,610,136]);   
    set(gcf(), 'MenuBar', 'none');
end


%%
% megallasi feltetelek
% epoch = hanyszor mutattuk be a tanitasi halmazt a neuronhalonak
function stop = stoppingCondition(E, epoch)
    br=0;
    if epoch > 10000
        stop = true;
        br=1;
        return
    end
    
    % halozat globalis hibajanak a valtozasi merteket teszteljuk a 10. ciklus
    % utan
    if length(E) < 10
        stop = false;
        br = 2;
        return
    end
    
    %ha mar van 10 hiba minta akkor mindig csak az utolsokat veszzuk
    %figyelembe
    if E(end-9) < E(end) || E(end-9) - E(end) < 1e-3
        stop = true;
        br = 3;
        return
    end
    
    stop = false;
end



%%
function [w,E] = offlineLearning(x, d, f, gradf, lr, stoppingCondition)
    % x dimenzioja
    [~, n] =size(x);
    
    % a sulyokat random modon inicializaljuk
    w =randn(n,size(d,2));
    
    %az tanitasi alkalmak szama
    epoch = 0;
    
    %a renszer hibajat tarolja
    E=[];
    
    while true
        %a sulyok es a bemenetek fuggvenyeben az ertek
        v = x * w;
        
        %kiszamitjuk az aktivacios fuggveny erteket
        y = f(v);
        
        %kiszamitjuk a hibat
        e = y - d;  %using squareloss function
        
        %irany szorozva a lepes hosszaval szorozva a bemenet derivaltjaval
        g = x' * (e .* gradf(v));   
        
        %a sulyzok ertekeit frissitjuk
        w = w - lr * g;
        
        %negyzetes hibafuggvenyt alkalmazunk
        E = [E; sum(e(:).^2)];
        
        % kilepesi feltetel
        %ha lejar a maximalis iteracio szam 
        %vagy
        %a hiba lecsokkent a kuszob ala
        if stoppingCondition(E, epoch)
            break; 
        end
        
        % iteraciok szamat noveljuk
        epoch = epoch + 1;
    end
end