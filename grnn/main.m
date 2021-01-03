clear all;

file = readtable('data.xlsx');
t = file{:,1}';
x = file{:,2}';
clear file;

[trainInd,testInd] = dividerand(size(x,2),0.7,0.3,0);

trainX=t(trainInd);
testX=t(testInd);

trainY=x(trainInd);
testY=x(testInd);

spread = 0.02;
net = newgrnn(trainX,trainY,spread);
% display(net);
% view(net);

net = init(net);

P=[trainX,testX];
T=[trainY,testY];

xr=sim(net,trainX);
xv=sim(net,testX);

figure;
gr=plot(trainX,trainY); grid;
set(gr(1),'LineStyle','.','Color','r','LineWidth',2);
legend('Исходные данные');
xlabel('Время'); ylabel('Значение');

figure;
gr=plot(trainX,trainY,trainX,xr,testX,xv); grid;
set(gr(1),'LineStyle','.','Color','r','LineWidth',2);
set(gr(2),'LineStyle','.','Color','g','LineWidth',2);
set(gr(3),'LineStyle','.','Color','b','LineWidth',2);
legend('Исходные данные','Аппроксимация обучающей выборки','Аппроксимация тестовой выборки');
xlabel('Время'); ylabel('Значение');

err1 = (trainY-xr);
err2 = (testY-xv);
figure;
gr=plot(trainX,err1,'.',testX,err2,'o'); grid;
set(gr(1),'LineStyle','.','Color','g','LineWidth',2);
set(gr(2),'LineStyle','.','Color','b','LineWidth',2);
legend('Ошибка обучающей выборки','Ошибка тестовой выборки');
xlabel('Время'); ylabel('Погрешность');

E = [err1,err2];
X = x;
SSE = sum(E.^2);
SSyy = sum((X-mean(X)).^2);
R2 = 1-SSE./SSyy; % коэффициент множественной детерминации
MSE = mean((E).^2); % средний квадрат ошибки
RMSE = sqrt(MSE); % средняя квадратичная ошибка
MAE = mean(abs(E)); % средняя абсолютная ошибка
MAPE = mean(abs(E)./X)*100; % средняя относительная ошибка
