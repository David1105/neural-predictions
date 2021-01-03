clear all;

file = readtable('data.xlsx');
t = file{:,1}';
x = file{:,2}';
clear file;

testPart = 0.1;
len = length(t);
first_test = round(len*(1-testPart));
[trainInd,valInd] = divideind(size(x,2),1:first_test-1,first_test:len);

trainX = t(trainInd);
valX = t(valInd);

trainY = x(trainInd);
valY = x(valInd);

net = feedforwardnet(10, 'trainlm');
net = configure(net,[0,6],[-1,1]);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:first_test-1;
net.divideParam.valInd = first_test:len;
net.divideParam.testInd = [];

% display(net);
% view(net);

net = init(net);

P = [trainX,valX];
T = [trainY,valY];

net.trainParam.epochs = 1000;
net.trainParam.max_fail = 1000;
net.trainParam.min_grad = 1E-5;

[net,tr] = train(net,P,T);
xr = sim(net,trainX);
xv = sim(net,valX);

figure; subplot(211)
gr = plot(t,x,trainX,xr,valX,xv); grid;
set(gr(1),'LineStyle','-','Color','r','LineWidth',2);
set(gr(2),'LineStyle','-','Color',[0 0.8 0],'LineWidth',2);
set(gr(3),'LineStyle','-','Color','b','LineWidth',2);
xlabel('Время'); ylabel('Значение');

E = [xr, xv] - x;
legend('Эталон','Аппроксимация','Прогнозирование');
subplot(212)
plot(t,E,'b','LineWidth',2); grid;
legend('Ошибка обучения');
xlabel('Время'); ylabel('Погрешность');

X = x;
SSE = sum(E.^2);
SSyy = sum((X-mean(X)).^2);
R2 = 1 - SSE./SSyy; % коэффициент множественной детерминации
MSE = mean((E).^2); % MSE (средний квадрат ошибки)
RMSE = sqrt(MSE); % средняя квадратичная ошибка
MAE = mean(abs(E)); % средняя абсолютная ошибка
MAPE = mean(abs(E)./X)*100; % средняя относительная ошибка
