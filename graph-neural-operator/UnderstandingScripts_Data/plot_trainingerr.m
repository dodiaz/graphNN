%% Making plot that shows the best input parameters to the Graph NNs

both_train = importdata('UAI6_s121_m100_radius0.2train.txt');
smoothonly_train = importdata('UAI6_s121_m100_radius0.2train2.txt');
origonly_train = importdata('UAI6_s121_m100_radius0.2train1.txt');
nogradnosmooth_train = importdata('UAI6_s121_m100_radius0.2train4.txt');

both_test = importdata('UAI6_s121_m100_radius0.2test.txt');
smoothonly_test = importdata('UAI6_s121_m100_radius0.2test2.txt');
origonly_test = importdata('UAI6_s121_m100_radius0.2test1.txt');
nogradnosmooth_test = importdata('UAI6_s121_m100_radius0.2test4.txt');

thick = 3;

subplot(2,1,1)
plot(both_train,'LineWidth',thick)
hold on
plot(smoothonly_train,'LineWidth',thick)
plot(origonly_train,'LineWidth',thick)
plot(nogradnosmooth_train,'LineWidth',thick)
xlabel("Epoch")
ylabel("Training error")
legend("x, y, a, a_{smooth}, \nabla a_{smooth}", "x, y, a_{smooth}, \nabla a_{smooth}", "x, y, a, \nabla a_{smooth}", "x, y, a")
[hleg,~,~] = legend('show');
title(hleg,'Inputs to Neural Network')
hleg.Title.Visible = 'on';
hold off





subplot(2,1,2)
plot(both_test,'LineWidth',thick)
hold on
plot(smoothonly_test,'LineWidth',thick)
plot(origonly_test,'LineWidth',thick)
plot(nogradnosmooth_test,'LineWidth',thick)
xlabel("Epoch")
ylabel("Testing error")
legend("x, y, a, a_{smooth}, \nabla a_{smooth}", "x, y, a_{smooth}, \nabla a_{smooth}", "x, y, a, \nabla a_{smooth}", "x, y, a")
[hleg,~,~] = legend('show');
title(hleg,'Inputs to Neural Network')
hleg.Title.Visible = 'on';
hold off




%%
train_Data = importdata("Poisson_s101_m100_radius0.2_epoch20train_poisson.txt");
test_Data = importdata("Poisson_s101_m100_radius0.2_epoch20test_poisson.txt");

thick = 3;

plot(train_Data(1:21), 'LineWidth', thick)
hold on
plot(test_Data(1:21), 'LineWidth', thick)
xlabel("Epoch")
ylabel("Error")
legend("Training error", "Testing error")
hold off


