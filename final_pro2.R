# Shuting Wang   002251577
# Zhaoyi Wang    002253523
# Yukuan Hao     002253251
# Liang Gu       002253248
#**************************************
require(ggplot2)
data=read.csv('D:/BU/2020.590-Master Project/2/SAheart.data')
# 1.Normalize Function
f_nor=function(data){
  mu=mean(data)
  sigma=sd(data)
  res=as.matrix((data-mu)/sigma)
  return(res)
}
# Get normalize x
nor_data=as.data.frame(apply(data[,c(2:5,7:10)], 2, f_nor))
nor_data=cbind(data$row.names,nor_data,data$chd)

# Set variable X and 'chd' column as Y
X=as.matrix(nor_data[1:100,3])
X=cbind(rep(1,nrow(X)),X)
Y=as.matrix(nor_data[1:100,10])

# f_w: w=x*beta which used in Sigmoid(w)
# f_log: Log-likelihood function(Objective function)
# f_sigmoid: Sigmoid function
# f_gradient: Gradient function
f_w=function(X,Beta){return(X%*%Beta)} 
f_log=function(W){
  # res= t(Y)%*%W-sum(log(1+exp(W)))
    res=sum(Y*W-log(1+exp(W)))
  return(res)
}
f_sigmoid=function(w){
  return(1/(1+exp(-w)))
}
f_gradient=function(X,Y,Beta){
  gra=t(X)%*%(Y-f_sigmoid(f_w(X,Beta)))
  return(gra)
}

# 2.Get log-beta
Beta=as.matrix(rep(0,ncol(X)))
# log_beta0=f_log(f_w(X,Beta))

# 3.Initialize parameter
eta=0.00001
epsilon=0.0000001
# 4.Gradient gra_0=gra(beta0,beta1)=14.5
temp=matrix()
for(i in 1:50000){
  w=X%*%Beta
  temp[i]=f_log(w)
  gra=f_gradient(X,Y,Beta)
  # temp[i]=gra[2]
  minus=eta*gra
  Beta=Beta+minus
# 5.Update gradient value
  if(all(abs(minus)<epsilon)){
    cat('loop time: ',i)
    break
  }
}
print(Beta)
# 6.Complete gradient ascent
cat('gra',gra,'Beta',Beta)
plot(temp,col='blue',main = 'Log-Beta')

# 7.Predict the labels for the test from row 101-462
X=as.matrix(nor_data[101:462,4])
X=cbind(rep(1,nrow(X)),X)
Y=as.matrix(nor_data[101:462,10])
W=X%*%Beta

# Test function
f_vector=function(w){
  if(f_sigmoid(w)>0.5){
    return(1.0) }
  else{ return(0) }
}

test=as.matrix(apply(W,1,f_vector))
test=cbind(Y,test)
temp=subset(test,test[,1]!=test[,2])
rate=(1-sum(temp[,1]+temp[,2])/nrow(test))*100
cat('Prediction: ',rate,'%')

# Plot
temp=as.data.frame(cbind(rep(1:nrow(X)),Y,f_sigmoid(W)))
names(temp)=c('Index','Label','Sigmoid')
ggplot(temp,aes(x=Index,y=Sigmoid,colour=Label))+geom_point()

# Find best prediction
i=matrix()
best_sigmoid=0.5
for (I in 1:40) {
  f_vector=function(w){
    if(f_sigmoid(w)>best_sigmoid){
      return(1.0) }
    else{ return(0) }
  }
  test=as.matrix(apply(W,1,f_vector))
  test=cbind(Y,test)
  temp=subset(test,test[,1]!=test[,2])
  rate=(1-sum(temp[,1]+temp[,2])/nrow(test))*100
  i[I]=rate
  best_sigmoid=best_sigmoid+0.01
  }
  cat('The best prediction:',max(i),'%')
  i=as.data.frame(i)
  i=cbind(rep(1:40),i)
  names(i)=c('a','b')
  ggplot(i,aes(x=a,y=b))+geom_line(linetype='dashed',col='gray',size=1)+
    geom_point(size=4,shape=21,col='darkred',fill='pink')+
    labs(x='Sigmoid',y='Predict',title=sprintf('The best predict value is %f%%',max(i)))
# write.csv(temp, file = "prediction.csv", row.names = FALSE)
  