# 使用官方OpenJDK 17运行时作为基础镜像
FROM openjdk:17-jdk-slim

# 设置工作目录
WORKDIR /app

# 将构建好的JAR文件复制到容器中
COPY build/libs/Gomoku-0.0.1-SNAPSHOT.jar app.jar

# 暴露Spring Boot默认端口
EXPOSE 8080

# 设置JVM参数（可选，用于优化性能）
ENV JAVA_OPTS="-Xmx512m -Xms256m"

# 运行应用
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
